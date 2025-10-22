import os
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from models.clip_encoder import FrozenCLIPTextEmbedder
from evaluation import *


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=5)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Text conditioning arguments
parser.add_argument('--use_text_condition', type=eval, default=True, choices=[True, False])
parser.add_argument('--text_dim', type=int, default=512, help='Text embedding dimension from CLIP')
parser.add_argument('--captions_path', type=str, default='./data/chairs_only.csv', help='Path to captions CSV file')
parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32', help='CLIP model version')
parser.add_argument('--use_alignment_loss', type=eval, default=True, choices=[True, False], help='Use text-shape alignment loss')
parser.add_argument('--alignment_weight', type=float, default=0.1, help='Weight for alignment loss (lambda_align)')

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--val_batch_size', type=int, default=32)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=20*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=40*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=10*THOUSAND)
parser.add_argument('--test_size', type=int, default=400)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file or log directory to resume training')
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets...')

# Use ShapeNetCoreText for text-conditioned training to ensure strict caption matching
from utils.dataset import ShapeNetCoreText
train_dset = ShapeNetCoreText(
    path=args.dataset_path,
    cates=args.categories,
    split='train',
    scale_mode=args.scale_mode,
    captions_path=args.captions_path,
)
val_dset = ShapeNetCoreText(
    path=args.dataset_path,
    cates=args.categories,
    split='val',
    scale_mode=args.scale_mode,
    captions_path=args.captions_path,
    )

# Custom collate function to handle text captions
def collate_fn(batch):
    """Custom collate function to handle string captions and matching status"""
    pointclouds = torch.stack([item['pointcloud'] for item in batch])
    captions = [item['caption'] for item in batch] if 'caption' in batch[0] else None
    cates = [item['cate'] for item in batch]
    ids = [item['id'] for item in batch]
    model_ids = [item.get('model_id', None) for item in batch]
    caption_matched = [item.get('caption_matched', False) for item in batch]

    result = {
        'pointcloud': pointclouds,
        'cate': cates,
        'id': ids,
        'model_id': model_ids,
        'caption_matched': caption_matched,
    }
    if captions is not None:
        result['caption'] = captions

    return result

train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
    collate_fn=collate_fn,
))

# Model
logger.info('Building model...')
if args.model == 'gaussian':
    model = GaussianVAE(args).to(args.device)
elif args.model == 'flow':
    model = FlowVAE(args).to(args.device)
logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)

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

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate and test
def train(it, pbar=None):
    # Load data
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)

    # Get text embeddings if using text conditioning
    text_emb = None
    if args.use_text_condition and text_encoder is not None:
        captions = batch.get('caption', None)  # Already a list of strings from collate_fn
        # Validate captions are all strings
        if captions is not None and isinstance(captions, list):
            # Filter out None values and ensure all are strings
            captions = [str(c) if c is not None else "a generic object" for c in captions]
            with torch.no_grad():
                text_emb = text_encoder(captions)
        else:
            # Fallback: create default captions
            batch_size = x.size(0)
            captions = ["a generic object"] * batch_size
            with torch.no_grad():
                text_emb = text_encoder(captions)

        # Log caption matching statistics every 100 iterations
        if it % 100 == 0:
            caption_matched_flags = batch.get('caption_matched', None)
            if caption_matched_flags is not None:
                matched_count = sum(caption_matched_flags)
                total_count = len(caption_matched_flags)
                match_rate = matched_count / total_count * 100 if total_count > 0 else 0
                writer.add_scalar('train/caption_match_rate', match_rate, it)
                if it % 1000 == 0:
                    logger.info(f'[Train] Caption matching: {matched_count}/{total_count} ({match_rate:.1f}%)')

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight
    loss = model.get_loss(x, kl_weight=kl_weight, text_emb=text_emb, writer=writer, it=it)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    # Update progress bar
    if pbar is not None:
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'grad': f'{orig_grad_norm:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        pbar.update(1)

    # Log to file occasionally
    if it % 100 == 0:
        logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
            it, loss.item(), orig_grad_norm, kl_weight
        ))

    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/kl_weight', kl_weight, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_inspect(it):
    """
    Validation: Generate samples with fixed captions and matching GT
    """
    model.eval()

    # # Fixed validation captions (3 samples)
    # fixed_captions = [
    #     "A brown coloured wooden chair with four legs and no side arms",
    #     "A wooden and metal baby chair. It is in light yellow color having black lining on seat and back support.",
    #     "this is wheel chair this is used in office."
    # ]


    num_vis_samples = 3
    real_samples, real_captions = [], []

    # # Find matching GT samples for each caption
    # for target_caption in fixed_captions:
    #     # Search in validation set first
    #     for data in train_dset:
    #         if data['caption'] == target_caption:
    #             real_samples.append(data['pointcloud'].unsqueeze(0))
    #             real_captions.append(data['caption'])
    #             break
    #     else:
    #         print(f"[Warning] Caption not found in training set: {target_caption}")
    #         continue
    import random

    val_indices = random.sample(range(len(train_dset)), 3)
    for i in val_indices:
        data = train_dset[i]
        real_samples.append(data['pointcloud'].unsqueeze(0))
        real_captions.append(data['caption'])

    x_real = torch.cat(real_samples, dim=0).to(args.device)

    # Generate with text conditioning
    z = torch.randn([num_vis_samples, args.latent_dim]).to(args.device)
    text_emb = None

    if args.use_text_condition and text_encoder is not None:
        with torch.no_grad():
            text_emb = text_encoder(real_captions)

    with torch.no_grad():
        x_gen = model.sample(z, args.sample_num_points, flexibility=args.flexibility, text_emb=text_emb)

    # Compute CD and EMD metrics
    from evaluation import EMD_CD
    metrics = EMD_CD(x_gen, x_real, args.val_batch_size, reduced=True)
    avg_cd = metrics['MMD-CD'].item()
    avg_emd = metrics['MMD-EMD'].item()

    # Log metrics
    logger.info(f'[Val {it}] CD: {avg_cd:.6f} | EMD: {avg_emd:.6f}')
    writer.add_scalar('val/chamfer_distance', avg_cd, it)
    writer.add_scalar('val/earth_mover_distance', avg_emd, it)

    # TensorBoard visualization (all 3 samples)
    caption_labels = ["WheelChair", "Chair", "BrownChair"]
    for i in range(num_vis_samples):
        writer.add_mesh(f'val/GT_{caption_labels[i]}', x_real[i:i+1], global_step=it)
        writer.add_mesh(f'val/Generated_{caption_labels[i]}', x_gen[i:i+1], global_step=it)
        writer.add_text(f'val/Caption_{caption_labels[i]}', real_captions[i], global_step=it)

    writer.flush()


def test(it):
    ref_pcs = []
    for i, data in enumerate(val_dset):
        if i >= args.test_size:
            break
        ref_pcs.append(data['pointcloud'].unsqueeze(0))
    ref_pcs = torch.cat(ref_pcs, dim=0)

    gen_pcs = []
    for i in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
        with torch.no_grad():
            z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
            text_emb = None

            # Use actual captions from validation set for diverse text-conditioned generation
            if args.use_text_condition and text_encoder is not None:
                # Get diverse captions from the validation set
                batch_start = i * args.val_batch_size
                batch_captions = []
                for j in range(args.val_batch_size):
                    idx = (batch_start + j) % len(val_dset)  # Wrap around if needed
                    data = val_dset[idx]
                    batch_captions.append(data.get('caption', 'a generic object'))

                with torch.no_grad():
                    text_emb = text_encoder(batch_captions)

                if i == 0:  # Log once
                    logger.info(f"[Test] Using text conditioning with diverse captions from val_dset")
                    logger.info(f"[Test] Example captions: {batch_captions[:3]}")

            # Generate with text conditioning
            x = model.sample(z, args.sample_num_points, flexibility=args.flexibility, text_emb=text_emb)
            gen_pcs.append(x.detach().cpu())
    gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]

    # Denormalize point clouds, all shapes have zero mean.
    # [WARNING]: Do NOT denormalize!
    # ref_pcs *= val_dset.stats['std']
    # gen_pcs *= val_dset.stats['std']

    with torch.no_grad():
        results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.val_batch_size)
        results = {k:v.item() for k, v in results.items()}
        jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
        results['jsd'] = jsd

    # CD related metrics
    writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], global_step=it)
    writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], global_step=it)
    writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], global_step=it)
    # EMD related metrics
    # writer.add_scalar('test/Coverage_EMD', results['lgan_cov-EMD'], global_step=it)
    # writer.add_scalar('test/MMD_EMD', results['lgan_mmd-EMD'], global_step=it)
    # writer.add_scalar('test/1NN_EMD', results['1-NN-EMD-acc'], global_step=it)
    # JSD
    writer.add_scalar('test/JSD', results['jsd'], global_step=it)

    # logger.info('[Test] Coverage  | CD %.6f | EMD %.6f' % (results['lgan_cov-CD'], results['lgan_cov-EMD']))
    # logger.info('[Test] MinMatDis | CD %.6f | EMD %.6f' % (results['lgan_mmd-CD'], results['lgan_mmd-EMD']))
    # logger.info('[Test] 1NN-Accur | CD %.6f | EMD %.6f' % (results['1-NN-CD-acc'], results['1-NN-EMD-acc']))
    logger.info('[Test] Coverage  | CD %.6f | EMD n/a' % (results['lgan_cov-CD'], ))
    logger.info('[Test] MinMatDis | CD %.6f | EMD n/a' % (results['lgan_mmd-CD'], ))
    logger.info('[Test] 1NN-Accur | CD %.6f | EMD n/a' % (results['1-NN-CD-acc'], ))
    logger.info('[Test] JsnShnDis | %.6f ' % (results['jsd']))

# Main loop
logger.info('Start training...')
try:
    it = 1
    with tqdm(total=args.max_iters, desc='Training', unit='iter',
              dynamic_ncols=True, leave=True) as pbar:
        while it <= args.max_iters:
            train(it, pbar)
            if it % args.val_freq == 0 or it == args.max_iters:
                pbar.write(f'\n[Iter {it}] Running validation...')
                validate_inspect(it)
                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
                pbar.write(f'[Iter {it}] Validation complete and checkpoint saved.\n')
            if it % args.test_freq == 0 or it == args.max_iters:
                pbar.write(f'\n[Iter {it}] Running test...')
                test(it)
                pbar.write(f'[Iter {it}] Test complete.\n')
            it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
