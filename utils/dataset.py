import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import pandas as pd
from tqdm.auto import tqdm


synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetCore(Dataset):

    GRAVITATIONAL_AXIS = 1

    def __init__(self, path, cates, split, scale_mode, transform=None, captions_path=None, modelid_mapping_path=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform
        self.captions_path = captions_path
        self.modelid_mapping_path = modelid_mapping_path if modelid_mapping_path else './data/modelid_mapping.json'

        self.pointclouds = []
        self.stats = None
        self.captions_dict = {}
        self.modelid_mapping = {}

        # Load modelId mapping if available
        if os.path.exists(self.modelid_mapping_path):
            self.load_modelid_mapping()

        # Load captions if path is provided
        if self.captions_path is not None and os.path.exists(self.captions_path):
            self.load_captions()

        self.get_statistics()
        self.load()

    def load_modelid_mapping(self):
        """Load modelId mapping from JSON file"""
        import json
        with open(self.modelid_mapping_path, 'r') as f:
            self.modelid_mapping = json.load(f)

    def load_captions(self):
        """Load captions from CSV file and create a dictionary mapping modelId to description"""
        df = pd.read_csv(self.captions_path)
        # Create dictionary: modelId -> description
        for _, row in df.iterrows():
            model_id = row['modelId']
            description = row['description']
            # Store all captions for each modelId (some models may have multiple captions)
            if model_id not in self.captions_dict:
                self.captions_dict[model_id] = []
            self.captions_dict[model_id].append(description)

    def get_statistics(self):

        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path, weights_only=False)
            return self.stats

        with h5py.File(self.path, 'r') as f:
            pointclouds = []
            for synsetid in self.cate_synsetids:
                for split in ('train', 'val', 'test'):
                    data = np.asarray(f[synsetid][split][...], dtype=np.float32)
                    pointclouds.append(torch.tensor(data))

        all_points = torch.cat(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):

        def _enumerate_pointclouds(f, synsetid, cate_name):
            """Enumerate point clouds with their modelId"""
            # Get modelId mapping for this synset and split
            mapping_key = f"{synsetid}_{self.split}"
            model_ids = []
            if mapping_key in self.modelid_mapping:
                model_ids = self.modelid_mapping[mapping_key].get('available_model_ids', [])

            for j, pc in enumerate(f[synsetid][self.split]):
                pc_array = np.asarray(pc, dtype=np.float32)
                # Get corresponding modelId (if available)
                model_id = model_ids[j] if j < len(model_ids) else None
                yield torch.tensor(pc_array), j, cate_name, model_id

        with h5py.File(self.path, mode='r') as f:
            for synsetid in self.cate_synsetids:
                cate_name = synsetid_to_cate[synsetid]

                for pc, pc_id, cate_name, model_id in _enumerate_pointclouds(f, synsetid, cate_name):

                    if self.scale_mode == 'global_unit':
                        shift = pc.mean(dim=0).reshape(1, 3)
                        scale = self.stats['std'].reshape(1, 1)
                    elif self.scale_mode == 'shape_unit':
                        shift = pc.mean(dim=0).reshape(1, 3)
                        scale = pc.flatten().std().reshape(1, 1)
                    elif self.scale_mode == 'shape_half':
                        shift = pc.mean(dim=0).reshape(1, 3)
                        scale = pc.flatten().std().reshape(1, 1) / (0.5)
                    elif self.scale_mode == 'shape_34':
                        shift = pc.mean(dim=0).reshape(1, 3)
                        scale = pc.flatten().std().reshape(1, 1) / (0.75)
                    elif self.scale_mode == 'shape_bbox':
                        pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                        pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                        shift = ((pc_min + pc_max) / 2).view(1, 3)
                        scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                    else:
                        shift = torch.zeros([1, 3])
                        scale = torch.ones([1, 1])

                    pc = (pc - shift) / scale

                    self.pointclouds.append({
                        'pointcloud': pc,
                        'cate': cate_name,
                        'id': pc_id,
                        'model_id': model_id,
                        'shift': shift,
                        'scale': scale
                    })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)

        # Add caption if available
        if self.captions_dict:
            model_id = data.get('model_id', None)
            cate = data['cate']

            # Try to get caption from exact modelId match first
            if model_id and model_id in self.captions_dict:
                # Get one of the captions for this specific model (randomly if multiple)
                captions = self.captions_dict[model_id]
                caption = random.choice(captions)
                data['caption'] = caption
                data['caption_matched'] = True  # Flag to indicate exact match
            else:
                # Fallback: use any caption from the same category
                # This should rarely happen with 98%+ coverage
                available_captions = []
                for mid, captions in self.captions_dict.items():
                    available_captions.extend(captions)

                if available_captions:
                    caption = random.choice(available_captions)
                    data['caption'] = caption
                    data['caption_matched'] = False  # Flag to indicate fallback
                else:
                    data['caption'] = f"a {cate}"
                    data['caption_matched'] = False
        else:
            data['caption'] = f"a generic object"
            data['caption_matched'] = False

        return data


class ShapeNetCoreText(ShapeNetCore):
    """
    Text-conditioned version of ShapeNetCore dataset.
    Only includes samples that have matching captions (caption_matched=True).
    Ensures strict alignment between point clouds and their text descriptions.
    """

    def __init__(self, path, cates, split, scale_mode, transform=None, captions_path=None, modelid_mapping_path=None):
        super().__init__(path, cates, split, scale_mode, transform, captions_path, modelid_mapping_path)

        # Filter to only keep samples with matched captions
        if captions_path is not None:
            self._filter_matched_captions()

    def _filter_matched_captions(self):
        """Filter dataset to only include samples with matched captions"""
        filtered_pointclouds = []

        for pc_data in self.pointclouds:
            model_id = pc_data.get('model_id', None)
            # Only keep if we have a valid model_id and it exists in captions_dict
            if model_id and model_id in self.captions_dict:
                filtered_pointclouds.append(pc_data)

        original_count = len(self.pointclouds)
        self.pointclouds = filtered_pointclouds
        filtered_count = len(self.pointclouds)

        print(f"[ShapeNetCoreText] Filtered {original_count} -> {filtered_count} samples "
              f"({filtered_count/original_count*100:.1f}% with matched captions)")

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)

        # Always get caption from exact modelId match (guaranteed to exist after filtering)
        model_id = data.get('model_id', None)
        cate = data['cate']

        if model_id and model_id in self.captions_dict:
            # Randomly select one caption if multiple exist
            captions = self.captions_dict[model_id]
            caption = random.choice(captions)
            data['caption'] = caption
            data['caption_matched'] = True
        else:
            # This should never happen after filtering, but add fallback for safety
            data['caption'] = f"a {cate}"
            data['caption_matched'] = False

        return data

