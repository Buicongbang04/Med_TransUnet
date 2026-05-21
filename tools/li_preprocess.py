import os
import re
import glob
import json
import random
import shutil
import argparse

import cv2
import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


DEFAULT_HU_MIN = -200
DEFAULT_HU_MAX = 250


# =========================================================
# Utility
# =========================================================

def get_case_id(path):
    base = os.path.basename(path)
    base = base.replace('.nii.gz', '').replace('.nii', '')

    match = re.search(r'(\d+)$', base)

    if match is None:
        return base

    return match.group(1)


# =========================================================
# Find image-label pairs
# =========================================================

def find_lits_pairs(root_dir):
    image_patterns = [
        os.path.join(root_dir, 'volume-*.nii'),
        os.path.join(root_dir, 'volume-*.nii.gz'),
        os.path.join(root_dir, '**', 'volume-*.nii'),
        os.path.join(root_dir, '**', 'volume-*.nii.gz'),
    ]

    label_patterns = [
        os.path.join(root_dir, 'segmentation-*.nii'),
        os.path.join(root_dir, 'segmentation-*.nii.gz'),
        os.path.join(root_dir, '**', 'segmentation-*.nii'),
        os.path.join(root_dir, '**', 'segmentation-*.nii.gz'),
    ]

    image_paths = []
    label_paths = []

    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern, recursive=True))

    for pattern in label_patterns:
        label_paths.extend(glob.glob(pattern, recursive=True))

    image_paths = sorted(set(image_paths))
    label_paths = sorted(set(label_paths))

    image_by_id = {get_case_id(path): path for path in image_paths}
    label_by_id = {get_case_id(path): path for path in label_paths}

    common_ids = sorted(
        set(image_by_id.keys()).intersection(label_by_id.keys()),
        key=lambda x: int(x) if x.isdigit() else x
    )

    pairs = []

    for case_id in common_ids:
        pairs.append({
            'case_id': case_id,
            'image_path': image_by_id[case_id],
            'label_path': label_by_id[case_id]
        })

    return pairs


# =========================================================
# Split dataset
# =========================================================

def split_cases(case_infos,
                split_file='',
                train_ratio=0.8,
                val_ratio=0.1,
                seed=42):

    case_by_id = {
        str(x['case_id']): x
        for x in case_infos
    }

    if split_file != '' and os.path.exists(split_file):

        with open(split_file, 'r') as f:
            split_info = json.load(f)['splits']

        def resolve_cases(case_names):
            resolved = []

            for item in case_names:
                raw = str(item)
                case_id = get_case_id(raw)

                if case_id not in case_by_id:
                    print(f'[WARN] Missing case: {raw}')
                    continue

                resolved.append(case_by_id[case_id])

            return resolved

        train_cases = resolve_cases(split_info.get('train', []))
        val_cases = resolve_cases(split_info.get('val', []))
        test_cases = resolve_cases(split_info.get('test', []))

        return train_cases, val_cases, test_cases

    random.seed(seed)

    shuffled = case_infos[:]
    random.shuffle(shuffled)

    n = len(shuffled)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_cases = shuffled[:n_train]
    val_cases = shuffled[n_train:n_train + n_val]
    test_cases = shuffled[n_train + n_val:]

    return train_cases, val_cases, test_cases


# =========================================================
# HU normalization
# =========================================================

def normalize_ct(volume,
                 hu_min=DEFAULT_HU_MIN,
                 hu_max=DEFAULT_HU_MAX):

    volume = volume.astype(np.float32)
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)

    return volume.astype(np.float32)


# =========================================================
# Keep strategy for train/val only
# =========================================================

def should_keep_slice(has_tumor,
                      has_liver,
                      liver_keep_prob=0.3,
                      empty_keep_prob=0.02):

    if has_tumor:
        return True

    if has_liver:
        return random.random() < liver_keep_prob

    return random.random() < empty_keep_prob


# =========================================================
# Resize
# =========================================================

def resize_slice(image,
                 label,
                 target_size=512):

    image = cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR
    )

    label = cv2.resize(
        label,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST
    )

    return image, label


# =========================================================
# Process train/val as 2D slice npz
# =========================================================

def process_trainval_case(case_info,
                          out_dir,
                          split,
                          target_size=512,
                          hu_min=DEFAULT_HU_MIN,
                          hu_max=DEFAULT_HU_MAX,
                          liver_keep_prob=0.3,
                          empty_keep_prob=0.02):

    case_id = str(case_info['case_id'])

    image_nii = nib.load(case_info['image_path'])
    label_nii = nib.load(case_info['label_path'])

    image = image_nii.get_fdata(dtype=np.float32)
    label = label_nii.get_fdata(dtype=np.float32)

    if image.shape != label.shape:
        print(f'[SKIP] Shape mismatch: {case_id}')
        return []

    image = normalize_ct(
        image,
        hu_min=hu_min,
        hu_max=hu_max
    )

    # LiTS17:
    # 0 = background
    # 1 = liver
    # 2 = tumor
    #
    # Binary tumor segmentation:
    # 0 = non-tumor
    # 1 = tumor
    tumor_mask = (label == 2).astype(np.uint8)

    saved_names = []
    depth = image.shape[2]

    for z in range(depth):

        img_slice = image[:, :, z]
        mask_slice = tumor_mask[:, :, z]
        liver_slice = label[:, :, z]

        has_tumor = mask_slice.sum() > 0
        has_liver = (liver_slice >= 1).sum() > 0

        keep = should_keep_slice(
            has_tumor=has_tumor,
            has_liver=has_liver,
            liver_keep_prob=liver_keep_prob,
            empty_keep_prob=empty_keep_prob
        )

        if not keep:
            continue

        img_slice, mask_slice = resize_slice(
            img_slice,
            mask_slice,
            target_size=target_size
        )

        filename = f'case_{case_id}_slice_{z:03d}.npz'
        save_path = os.path.join(out_dir, filename)

        np.savez_compressed(
            save_path,
            image=img_slice.astype(np.float32),
            label=mask_slice.astype(np.uint8),
            case_id=case_id,
            slice_index=z,
            original_shape=np.array(image.shape),
            original_spacing=np.array(image_nii.header.get_zooms()[:3])
        )

        saved_names.append(filename.replace('.npz', ''))

    return saved_names


# =========================================================
# Process test as full volume h5
# =========================================================

def process_test_case_as_volume(case_info,
                                out_dir,
                                target_size=512,
                                hu_min=DEFAULT_HU_MIN,
                                hu_max=DEFAULT_HU_MAX):

    case_id = str(case_info['case_id'])

    image_nii = nib.load(case_info['image_path'])
    label_nii = nib.load(case_info['label_path'])

    image = image_nii.get_fdata(dtype=np.float32)
    label = label_nii.get_fdata(dtype=np.float32)

    if image.shape != label.shape:
        print(f'[SKIP] Shape mismatch: {case_id}')
        return None

    original_shape = image.shape
    original_spacing = image_nii.header.get_zooms()[:3]

    image = normalize_ct(
        image,
        hu_min=hu_min,
        hu_max=hu_max
    )

    tumor_mask = (label == 2).astype(np.uint8)

    img_slices = []
    mask_slices = []

    depth = image.shape[2]

    for z in range(depth):

        img_slice = image[:, :, z]
        mask_slice = tumor_mask[:, :, z]

        img_slice, mask_slice = resize_slice(
            img_slice,
            mask_slice,
            target_size=target_size
        )

        img_slices.append(img_slice.astype(np.float32))
        mask_slices.append(mask_slice.astype(np.uint8))

    # TransUNet test format:
    # image: [D, H, W]
    # label: [D, H, W]
    image_3d = np.stack(img_slices, axis=0)
    label_3d = np.stack(mask_slices, axis=0)

    case_name = f'case_{case_id}'
    save_path = os.path.join(out_dir, f'{case_name}.npy.h5')

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('image', data=image_3d, compression='gzip')
        f.create_dataset('label', data=label_3d, compression='gzip')

        f.attrs['case_id'] = case_id
        f.attrs['original_shape'] = original_shape
        f.attrs['original_spacing'] = original_spacing
        f.attrs['target_size'] = target_size
        f.attrs['hu_min'] = hu_min
        f.attrs['hu_max'] = hu_max

    return case_name


# =========================================================
# Write txt list
# =========================================================

def write_txt(path, names):
    with open(path, 'w') as f:
        for name in names:
            f.write(name + '\n')


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--lits-root',
        type=str,
        default='data/LiTS17'
    )

    parser.add_argument(
        '--out-root',
        type=str,
        default='data/LiTS'
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=512
    )

    parser.add_argument(
        '--hu-min',
        type=float,
        default=DEFAULT_HU_MIN
    )

    parser.add_argument(
        '--hu-max',
        type=float,
        default=DEFAULT_HU_MAX
    )

    parser.add_argument(
        '--liver-keep-prob',
        type=float,
        default=0.3
    )

    parser.add_argument(
        '--empty-keep-prob',
        type=float,
        default=0.02
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )

    parser.add_argument(
        '--clean',
        action='store_true'
    )

    parser.add_argument(
        '--split-file',
        type=str,
        default='',
        help='Path to split json file'
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.clean and os.path.exists(args.out_root):
        shutil.rmtree(args.out_root)

    train_dir = os.path.join(args.out_root, 'train_npz')
    val_dir = os.path.join(args.out_root, 'val_npz')

    # Important:
    # TransUNet original test loader expects volume h5 files.
    test_dir = os.path.join(args.out_root, 'test_vol_h5')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    list_dir = os.path.join(args.out_root, 'lists_LiTS')
    os.makedirs(list_dir, exist_ok=True)

    case_infos = find_lits_pairs(args.lits_root)

    print(f'Found {len(case_infos)} cases')

    train_cases, val_cases, test_cases = split_cases(
        case_infos,
        split_file=args.split_file,
        seed=args.seed
    )

    print(f'Train cases: {len(train_cases)}')
    print(f'Val cases:   {len(val_cases)}')
    print(f'Test cases:  {len(test_cases)}')

    splits = {
        'train': (train_cases, train_dir),
        'val': (val_cases, val_dir),
        'test': (test_cases, test_dir)
    }

    split_names = {}

    for split_name, (cases, out_dir) in splits.items():

        print(f'\nProcessing {split_name}...')

        all_names = []

        for case_info in tqdm(cases):

            if split_name == 'test':

                name = process_test_case_as_volume(
                    case_info=case_info,
                    out_dir=out_dir,
                    target_size=args.img_size,
                    hu_min=args.hu_min,
                    hu_max=args.hu_max
                )

                if name is not None:
                    all_names.append(name)

            else:

                names = process_trainval_case(
                    case_info=case_info,
                    out_dir=out_dir,
                    split=split_name,
                    target_size=args.img_size,
                    hu_min=args.hu_min,
                    hu_max=args.hu_max,
                    liver_keep_prob=args.liver_keep_prob,
                    empty_keep_prob=args.empty_keep_prob
                )

                all_names.extend(names)

        split_names[split_name] = all_names

        if split_name == 'test':
            print(f'{split_name}: {len(all_names)} full volumes')
        else:
            print(f'{split_name}: {len(all_names)} slices')

    write_txt(
        os.path.join(list_dir, 'train.txt'),
        split_names['train']
    )

    write_txt(
        os.path.join(list_dir, 'val.txt'),
        split_names['val']
    )

    write_txt(
        os.path.join(list_dir, 'test.txt'),
        split_names['test']
    )

    print('\nDone.')
    print(f'Dataset saved to: {args.out_root}')
    print(f'Train npz:        {train_dir}')
    print(f'Val npz:          {val_dir}')
    print(f'Test volume h5:   {test_dir}')
    print(f'Lists:            {list_dir}')


if __name__ == '__main__':
    main()

