"""
Visualize generated samples from training captions and compare with GT
This script:
1. Loads training dataset with captions
2. Generates point clouds from training captions
3. Compares generated samples with GT using CD/EMD metrics
4. Saves visualizations as images
"""

import os
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.dataset import ShapeNetCoreText
from utils.misc import seed_all
from models.vae_gaussian import GaussianVAE
from models.vae_flow import FlowVAE
from models.clip_encoder import FrozenCLIPTextEmbedder
from evaluation.evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets, EMD_CD


def visualize_point_cloud(ax, points, title, color='blue', elev=30, azim=45):
    """
    Visualize a single point cloud on given axis

    Args:
        ax: matplotlib 3D axis
        points: (N, 3) point cloud
        title: title for the plot
        color: color of points
        elev: elevation angle
        azim: azimuth angle
    """
    points = points.cpu().numpy() if torch.is_tensor(points) else points

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=1, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio
    max_range = np.abs(points).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.view_init(elev=elev, azim=azim)


def normalize_point_cloud(pc, mode='shape_unit'):
    """
    Normalize point cloud

    Args:
        pc: (N, 3) point cloud tensor
        mode: normalization mode
    """
    if mode == 'shape_unit':
        shift = pc.mean(dim=0, keepdim=True)
        scale = pc.flatten().std()
    elif mode == 'shape_bbox':
        pc_max = pc.max(dim=0, keepdim=True)[0]
        pc_min = pc.min(dim=0, keepdim=True)[0]
        shift = (pc_min + pc_max) / 2
        scale = (pc_max - pc_min).max() / 2
    else:
        return pc

    pc = (pc - shift) / scale
    return pc


def main(args):
    # Set seed
    seed_all(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[INFO] Loading checkpoint from: {args.ckpt}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)

    # Determine model type from checkpoint
    if 'latent_flow_depth' in ckpt['args']:
        print("[INFO] Model type: FlowVAE")
        model = FlowVAE(ckpt['args']).to(args.device)
    else:
        print("[INFO] Model type: GaussianVAE")
        model = GaussianVAE(ckpt['args']).to(args.device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print(f"[INFO] Loaded checkpoint from iteration {ckpt.get('iteration', 'unknown')}")

    # Load CLIP text encoder if using text conditioning
    use_text = ckpt['args'].use_text_condition if hasattr(ckpt['args'], 'use_text_condition') else False
    if use_text:
        print("[INFO] Loading CLIP text encoder...")
        clip_model = ckpt['args'].clip_model if hasattr(ckpt['args'], 'clip_model') else 'openai/clip-vit-base-patch32'
        text_encoder = FrozenCLIPTextEmbedder(version=clip_model, device=args.device, return_sequence=True)
        text_encoder = text_encoder.to(args.device)  # Move to GPU
        text_encoder.eval()
    else:
        text_encoder = None
        print("[WARNING] Model was trained without text conditioning!")

    # Use custom captions or load dataset
    if args.custom_captions is not None:
        print(f"[INFO] Using {len(args.custom_captions)} custom captions")
        all_captions = args.custom_captions
        num_samples = len(all_captions)
        use_dataset = False
    else:
        # Load training dataset
        print(f"[INFO] Loading dataset from: {args.dataset_path}")
        train_dataset = ShapeNetCoreText(
            path=args.dataset_path,
            cates=args.categories,
            split='test',
            scale_mode=args.normalize,
            captions_path=args.captions_path,
        )

        print(f"[INFO] Dataset size: {len(train_dataset)}")
        # ShapeNetCoreText already filters to only caption-matched samples
        print(f"[INFO] All samples have matching captions (pre-filtered by dataset)")

        # Select samples to visualize
        num_samples = min(args.num_samples, len(train_dataset))

        # Sample indices (evenly spaced or random)
        if args.sample_mode == 'random':
            indices = np.random.choice(len(train_dataset), num_samples, replace=False)
        else:  # 'evenly'
            indices = np.linspace(0, len(train_dataset) - 1, num_samples, dtype=int)

        use_dataset = True

    print(f"[INFO] Generating {num_samples} samples...")

    # Store results
    all_gt = []
    all_generated = []
    caption_list = []
    all_cd_distances = []
    all_emd_distances = []

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Generating samples"):
            if use_dataset:
                # Get GT data from dataset
                data = train_dataset[indices[i]]
                gt_pc = data['pointcloud'].to(args.device)  # (N, 3)
                caption = data.get('caption', 'No caption')
                model_id = data.get('model_id', 'unknown')
            else:
                # Using custom captions - no GT available
                gt_pc = None
                caption = all_captions[i]

            # Encode text
            if text_encoder is not None and caption != 'No caption':
                text_emb = text_encoder([caption])
            else:
                text_emb = None

            # Sample from model
            # Note: We sample random z from prior, not encoding GT
            # This tests the model's ability to generate from text + random latent
            latent_dim = ckpt['args'].latent_dim
            z = torch.randn(1, latent_dim).to(args.device)

            generated_pc = model.sample(
                z,
                num_points=args.sample_num_points,
                flexibility=args.flexibility,
                text_emb=text_emb
            )  # (1, N, 3)

            generated_pc = generated_pc.squeeze(0)  # (N, 3)

            # Compute metrics only if GT is available
            if gt_pc is not None:
                # Ensure same number of points for fair comparison
                if gt_pc.shape[0] != generated_pc.shape[0]:
                    # Resample GT to match generated
                    perm = torch.randperm(gt_pc.shape[0])[:generated_pc.shape[0]]
                    gt_pc_resampled = gt_pc[perm]
                else:
                    gt_pc_resampled = gt_pc

                # Compute CD and EMD
                metrics = EMD_CD(
                    generated_pc.unsqueeze(0),
                    gt_pc_resampled.unsqueeze(0),
                    batch_size=1
                )

                cd_dist = metrics['MMD-CD'].item()
                emd_dist = metrics['MMD-EMD'].item()

                all_gt.append(gt_pc_resampled.cpu())
            else:
                cd_dist = None
                emd_dist = None

            # Store results
            all_generated.append(generated_pc.cpu())
            caption_list.append(caption)
            all_cd_distances.append(cd_dist)
            all_emd_distances.append(emd_dist)

    # Compute statistics (only if GT is available)
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    if use_dataset:
        # Filter out None values
        valid_cd = [cd for cd in all_cd_distances if cd is not None]
        valid_emd = [emd for emd in all_emd_distances if emd is not None]

        if valid_cd:
            print(f"Average Chamfer Distance (CD):  {np.mean(valid_cd):.6f} ± {np.std(valid_cd):.6f}")
            print(f"Average Earth Mover Distance:    {np.mean(valid_emd):.6f} ± {np.std(valid_emd):.6f}")
            print(f"Min CD: {np.min(valid_cd):.6f}")
            print(f"Max CD: {np.max(valid_cd):.6f}")
    else:
        print("Custom captions used - no ground truth metrics available")
        print(f"Generated {num_samples} point clouds from custom captions")
    print("="*80)

    # Save statistics to file
    stats_file = os.path.join(args.save_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Mode: {'Dataset' if use_dataset else 'Custom captions'}\n\n")

        if use_dataset:
            valid_cd = [cd for cd in all_cd_distances if cd is not None]
            valid_emd = [emd for emd in all_emd_distances if emd is not None]
            if valid_cd:
                f.write(f"Average Chamfer Distance (CD):  {np.mean(valid_cd):.6f} ± {np.std(valid_cd):.6f}\n")
                f.write(f"Average Earth Mover Distance:    {np.mean(valid_emd):.6f} ± {np.std(valid_emd):.6f}\n")
                f.write(f"Min CD: {np.min(valid_cd):.6f}\n")
                f.write(f"Max CD: {np.max(valid_cd):.6f}\n")
        else:
            f.write("Custom captions used - no ground truth metrics available\n")

        f.write("="*80 + "\n\n")

        # Per-sample details
        f.write("PER-SAMPLE DETAILS\n")
        f.write("="*80 + "\n")
        for i, (caption, cd, emd) in enumerate(zip(caption_list, all_cd_distances, all_emd_distances)):
            f.write(f"Sample {i}:\n")
            f.write(f"  Caption: {caption}\n")
            if cd is not None:
                f.write(f"  CD:  {cd:.6f}\n")
                f.write(f"  EMD: {emd:.6f}\n")
            else:
                f.write(f"  CD:  N/A (custom caption)\n")
                f.write(f"  EMD: N/A (custom caption)\n")
            f.write("\n")

    print(f"[INFO] Statistics saved to: {stats_file}")

    # Visualize samples
    print(f"\n[INFO] Creating visualizations...")

    samples_per_page = args.samples_per_page
    num_pages = (num_samples + samples_per_page - 1) // samples_per_page

    for page in range(num_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, num_samples)
        page_samples = end_idx - start_idx

        if use_dataset:
            # Create figure with subplots (2 columns: GT and Generated)
            fig = plt.figure(figsize=(12, 4 * page_samples))

            for i in range(page_samples):
                sample_idx = start_idx + i
                caption = caption_list[sample_idx]
                cd = all_cd_distances[sample_idx]

                # GT point cloud
                ax1 = fig.add_subplot(page_samples, 2, 2*i + 1, projection='3d')
                visualize_point_cloud(
                    ax1,
                    all_gt[sample_idx],
                    f"Ground Truth #{sample_idx}\n\"{caption[:50]}...\"" if len(caption) > 50 else f"Ground Truth #{sample_idx}\n\"{caption}\"",
                    color='blue'
                )

                # Generated point cloud
                ax2 = fig.add_subplot(page_samples, 2, 2*i + 2, projection='3d')
                visualize_point_cloud(
                    ax2,
                    all_generated[sample_idx],
                    f"Generated from Caption #{sample_idx}\nCD: {cd:.4f}",
                    color='red'
                )
        else:
            # For custom captions, only show generated (no GT)
            fig = plt.figure(figsize=(12, 4 * page_samples))

            for i in range(page_samples):
                sample_idx = start_idx + i
                caption = caption_list[sample_idx]

                # Generated point cloud (single column layout)
                ax = fig.add_subplot(page_samples, 1, i + 1, projection='3d')
                visualize_point_cloud(
                    ax,
                    all_generated[sample_idx],
                    f"Generated #{sample_idx}\n\"{caption[:70]}...\"" if len(caption) > 70 else f"Generated #{sample_idx}\n\"{caption}\"",
                    color='red'
                )

        plt.tight_layout()

        # Save figure
        output_file = os.path.join(args.save_dir, f'comparison_page_{page+1}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Saved visualization page {page+1}/{num_pages}: {output_file}")

    # Create a summary figure with best and worst samples (only if using dataset)
    if use_dataset:
        print(f"\n[INFO] Creating best/worst samples visualization...")

        # Filter out None values and get valid indices
        valid_indices = [i for i, cd in enumerate(all_cd_distances) if cd is not None]
        valid_cd = [all_cd_distances[i] for i in valid_indices]

        if len(valid_indices) >= 3:
            # Sort by CD
            sorted_idx = np.argsort(valid_cd)
            best_indices = [valid_indices[i] for i in sorted_idx[:3]]  # Top 3 best
            worst_indices = [valid_indices[i] for i in sorted_idx[-3:]]  # Top 3 worst

            fig = plt.figure(figsize=(12, 16))

            for i, idx in enumerate(best_indices):
                caption = caption_list[idx]
                cd = all_cd_distances[idx]

                # GT
                ax1 = fig.add_subplot(6, 2, 2*i + 1, projection='3d')
                visualize_point_cloud(
                    ax1,
                    all_gt[idx],
                    f"BEST #{i+1} - Ground Truth\n\"{caption[:35]}...\"" if len(caption) > 35 else f"BEST #{i+1} - Ground Truth\n\"{caption}\"",
                    color='blue'
                )

                # Generated
                ax2 = fig.add_subplot(6, 2, 2*i + 2, projection='3d')
                visualize_point_cloud(
                    ax2,
                    all_generated[idx],
                    f"BEST #{i+1} - Generated\nCD: {cd:.4f}",
                    color='green'
                )

            for i, idx in enumerate(worst_indices):
                caption = caption_list[idx]
                cd = all_cd_distances[idx]

                # GT
                ax1 = fig.add_subplot(6, 2, 6 + 2*i + 1, projection='3d')
                visualize_point_cloud(
                    ax1,
                    all_gt[idx],
                    f"WORST #{i+1} - Ground Truth\n\"{caption[:35]}...\"" if len(caption) > 35 else f"WORST #{i+1} - Ground Truth\n\"{caption}\"",
                    color='blue'
                )

                # Generated
                ax2 = fig.add_subplot(6, 2, 6 + 2*i + 2, projection='3d')
                visualize_point_cloud(
                    ax2,
                    all_generated[idx],
                    f"WORST #{i+1} - Generated\nCD: {cd:.4f}",
                    color='orange'
                )

            plt.tight_layout()
            summary_file = os.path.join(args.save_dir, 'best_worst_comparison.png')
            plt.savefig(summary_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Saved best/worst comparison: {summary_file}")
        else:
            print(f"[WARNING] Not enough samples ({len(valid_indices)}) for best/worst comparison")
    else:
        print(f"\n[INFO] Skipping best/worst comparison (custom captions mode)")

    # Save point clouds as numpy arrays
    if args.save_pointclouds:
        print(f"\n[INFO] Saving point clouds...")
        pc_dir = os.path.join(args.save_dir, 'pointclouds')
        os.makedirs(pc_dir, exist_ok=True)

        for i in range(num_samples):
            np.save(
                os.path.join(pc_dir, f'sample_{i:04d}_gt.npy'),
                all_gt[i].numpy()
            )
            np.save(
                os.path.join(pc_dir, f'sample_{i:04d}_generated.npy'),
                all_generated[i].numpy()
            )

        print(f"[INFO] Point clouds saved to: {pc_dir}")

    print(f"\n[SUCCESS] All visualizations saved to: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize generated samples vs GT from training captions')

    # Model checkpoint
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')

    # Dataset
    parser.add_argument('--dataset_path', type=str,
                        default='./data/shapenet.hdf5',
                        help='Path to HDF5 dataset')
    parser.add_argument('--categories', type=str, nargs='+', default=['chair'],
                        help='Categories to visualize')
    parser.add_argument('--captions_path', type=str,
                        default='./data/captions.tablechair.csv',
                        help='Path to captions CSV file')

    # Sampling
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate and visualize')
    parser.add_argument('--sample_mode', type=str, default='random',
                        choices=['random', 'evenly'],
                        help='How to select samples from dataset')
    parser.add_argument('--custom_captions', type=str, nargs='+', default=None,
                        help='Custom captions to generate from (instead of using dataset)')
    parser.add_argument('--sample_num_points', type=int, default=2048,
                        help='Number of points to generate')
    parser.add_argument('--flexibility', type=float, default=0.0,
                        help='Flexibility for sampling')

    # Normalization
    parser.add_argument('--normalize', type=str, default='shape_unit',
                        choices=['shape_unit', 'shape_bbox', 'none'],
                        help='Point cloud normalization mode')

    # Output
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--samples_per_page', type=int, default=5,
                        help='Number of samples per visualization page')
    parser.add_argument('--save_pointclouds', action='store_true',
                        help='Save point clouds as numpy arrays')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed')

    args = parser.parse_args()

    main(args)
