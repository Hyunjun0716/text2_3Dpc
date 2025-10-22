import os
import time
import math
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from models.clip_encoder import FrozenCLIPTextEmbedder
from evaluation import *

def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
parser.add_argument('--categories', type=str_list, default=['chair'], help='Categories to test')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=64)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_unit', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
# Text conditioning
parser.add_argument('--use_text_condition', type=eval, default=True, choices=[True, False])
parser.add_argument('--text_prompt', type=str, default=None, help='Text prompt for conditional generation (if None, uses default)')
parser.add_argument('--captions_path', type=str, default='./data/captions.tablechair.csv', help='Path to captions CSV file')
parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32', help='CLIP model version')
args = parser.parse_args()


# Logging
save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt, weights_only=False)
seed_all(args.seed)

# Datasets and loaders
logger.info('Loading datasets...')

# Use ShapeNetCoreText if using text conditioning, otherwise use standard ShapeNetCore
if args.use_text_condition and args.captions_path:
    from utils.dataset import ShapeNetCoreText
    test_dset = ShapeNetCoreText(
        path=args.dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=args.normalize,
        captions_path=args.captions_path,
    )
    logger.info(f'Using ShapeNetCoreText dataset with captions from {args.captions_path}')
else:
    test_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=args.normalize,
    )
    logger.info('Using standard ShapeNetCore dataset (no text conditioning)')

test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args']).to(args.device)
logger.info(repr(model))
# if ckpt['args'].spectral_norm:
#     add_spectral_norm(model, logger=logger)
model.load_state_dict(ckpt['state_dict'])

# Text encoder (CLIP) - frozen
text_encoder = None
if args.use_text_condition:
    logger.info('Loading CLIP text encoder...')
    text_encoder = FrozenCLIPTextEmbedder(
        version=args.clip_model,
        device=args.device,
        max_length=77,
        return_sequence=True  # Return token sequence for cross attention
    )
    text_encoder = text_encoder.to(args.device)
    logger.info('CLIP text encoder loaded and frozen (returning token sequences for cross attention).')

# Reference Point Clouds
ref_pcs = []
for i, data in enumerate(test_dset):
    ref_pcs.append(data['pointcloud'].unsqueeze(0))
ref_pcs = torch.cat(ref_pcs, dim=0)

# Generate Point Clouds
gen_pcs = []

# Prepare text embeddings if using text conditioning
text_emb_single = None
if args.use_text_condition and text_encoder is not None:
    if args.text_prompt:
        # Use the provided text prompt for ALL generations
        logger.info(f'Using text prompt: "{args.text_prompt}"')
        with torch.no_grad():
            text_emb_single = text_encoder([args.text_prompt])  # Single prompt as list
    else:
        # Use a default prompt
        default_prompt = f"a {args.categories[0]}"
        logger.info(f'Using default prompt: "{default_prompt}"')
        with torch.no_grad():
            text_emb_single = text_encoder([default_prompt])  # Single prompt as list

for i in tqdm(range(0, math.ceil(len(test_dset) / args.batch_size)), 'Generate'):
    with torch.no_grad():
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)

        # Repeat text embedding for batch
        batch_text_emb = None
        if text_emb_single is not None:
            # text_emb_single shape: [1, seq_len, dim], repeat to [batch_size, seq_len, dim]
            batch_text_emb = text_emb_single.repeat(args.batch_size, 1, 1)

        x = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, text_emb=batch_text_emb)
        gen_pcs.append(x.detach().cpu())
gen_pcs = torch.cat(gen_pcs, dim=0)[:len(test_dset)]
if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)

# Save
logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'out.npy'), gen_pcs.numpy())

# Compute metrics
with torch.no_grad():
    results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.batch_size)
    results = {k:v.item() for k, v in results.items()}
    jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
