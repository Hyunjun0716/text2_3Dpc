"""
Build ShapeNet HDF5 dataset from raw .txt point cloud files
"""
import h5py
import numpy as np
import os
from tqdm import tqdm
from glob import glob

# Category mapping
synsetoffset2category = {
    '02691156': 'airplane', '02773838': 'bag', '02954340': 'cap',
    '02958343': 'car', '03001627': 'chair', '03261776': 'earphone',
    '03467517': 'guitar', '03624134': 'knife', '03636649': 'lamp',
    '03642806': 'laptop', '03790512': 'motorbike', '03797390': 'mug',
    '03948459': 'pistol', '04099429': 'rocket', '04225987': 'skateboard',
    '04379243': 'table',
}

def load_point_cloud(file_path, num_points=2048):
    """Load point cloud from .txt file and sample to fixed number of points"""
    points = np.loadtxt(file_path)

    # Extract only x,y,z coordinates (first 3 columns)
    if points.shape[1] > 3:
        points = points[:, :3]

    # Sample to fixed number of points
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        # Pad with duplicates if needed
        indices = np.random.choice(len(points), num_points, replace=True)
        points = points[indices]

    return points.astype(np.float32)

def load_train_test_split(data_dir, synsetid):
    """Load train/test split information"""
    split_file = os.path.join(data_dir, 'train_test_split', f'shuffled_{synsetoffset2category[synsetid]}_file_list.json')

    if os.path.exists(split_file):
        import json
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        return split_info
    else:
        return None

def build_hdf5_dataset(data_dir='./data', output_file='./data/shapenet.hdf5', num_points=2048):
    """Build HDF5 dataset from raw ShapeNet .txt files"""

    print(f"Building ShapeNet HDF5 dataset...")
    print(f"Output file: {output_file}")
    print(f"Points per shape: {num_points}")

    with h5py.File(output_file, 'w') as hf:
        for synsetid, category in synsetoffset2category.items():
            category_dir = os.path.join(data_dir, synsetid)

            if not os.path.exists(category_dir):
                print(f"Warning: Category directory not found: {category_dir}")
                continue

            print(f"\nProcessing category: {category} ({synsetid})")

            # Find all .txt files
            txt_files = sorted(glob(os.path.join(category_dir, '*.txt')))

            if len(txt_files) == 0:
                print(f"Warning: No .txt files found in {category_dir}")
                continue

            print(f"Found {len(txt_files)} point cloud files")

            # Load train/test split info
            split_info = load_train_test_split(data_dir, synsetid)

            # Categorize files into train/val/test
            train_files = []
            val_files = []
            test_files = []

            if split_info:
                train_ids = set(split_info.get('train', []))
                val_ids = set(split_info.get('val', []))
                test_ids = set(split_info.get('test', []))

                for f in txt_files:
                    file_id = os.path.basename(f).replace('.txt', '')
                    if file_id in train_ids:
                        train_files.append(f)
                    elif file_id in val_ids:
                        val_files.append(f)
                    elif file_id in test_ids:
                        test_files.append(f)
            else:
                # Default split: 70% train, 10% val, 20% test
                n_total = len(txt_files)
                n_train = int(0.7 * n_total)
                n_val = int(0.1 * n_total)

                train_files = txt_files[:n_train]
                val_files = txt_files[n_train:n_train + n_val]
                test_files = txt_files[n_train + n_val:]

            print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

            # Create group for this category
            grp = hf.create_group(synsetid)

            # Process each split
            for split_name, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
                if len(file_list) == 0:
                    grp.create_dataset(split_name, shape=(0, num_points, 3), dtype=np.float32)
                    continue

                point_clouds = []
                for f in tqdm(file_list, desc=f"  Loading {split_name}", ncols=80):
                    try:
                        pc = load_point_cloud(f, num_points)
                        point_clouds.append(pc)
                    except Exception as e:
                        print(f"Error loading {f}: {e}")

                if len(point_clouds) > 0:
                    point_clouds = np.stack(point_clouds, axis=0)
                    grp.create_dataset(split_name, data=point_clouds, dtype=np.float32)
                else:
                    grp.create_dataset(split_name, shape=(0, num_points, 3), dtype=np.float32)

    print(f"\nDataset saved to {output_file}")
    print(f"Total size: {os.path.getsize(output_file) / (1024**3):.2f} GB")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing raw ShapeNet files')
    parser.add_argument('--output', type=str, default='./data/shapenet.hdf5', help='Output HDF5 file path')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per shape')
    args = parser.parse_args()

    build_hdf5_dataset(args.data_dir, args.output, args.num_points)
