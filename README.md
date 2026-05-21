# ThesisProject26 TransUNet

PyTorch TransUNet code for medical image segmentation. This fork keeps the original Synapse workflow and adds a LiTS17 binary tumor segmentation pipeline with preprocessing, fold-based training, and fold-based testing scripts.

## Setup

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The training and testing scripts call `.cuda()`, so use a machine with an NVIDIA GPU and a CUDA-compatible PyTorch installation.

Place the [R50-ViT-B_16 pretrained weight](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/R50%2BViT-B_16.npz;tab=live_object?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))) here:

```text
pretrained/imagenet21k_R50+ViT-B_16.npz
```

That path is used by `networks/vit_seg_configs.py` for `--vit_name R50-ViT-B_16`.

## Data

For LiTS17, place raw NIfTI image/label pairs under:

```text
data/LiTS17/
  volume-0.nii
  segmentation-0.nii
  volume-1.nii
  segmentation-1.nii
  ...
```

Fold split files are expected under:

```text
data/splits/liver_fold_0.json
data/splits/liver_fold_1.json
data/splits/liver_fold_2.json
```

The current scripts run folds `0..2`. Each split JSON should contain `splits.train`, `splits.val`, and `splits.test`.

For Synapse, keep the original TransUNet preprocessed layout:

```text
../data/Synapse/train_npz/
../data/Synapse/test_vol_h5/
lists/lists_Synapse/
```

## Preprocess LiTS17

Preprocess one fold:

```bash
python tools/li_preprocess.py \
  --lits-root data/LiTS17 \
  --out-root data/LiTS/fold_0 \
  --img-size 512 \
  --split-file data/splits/liver_fold_0.json \
  --clean
```

This creates:

```text
data/LiTS/fold_0/train_npz/
data/LiTS/fold_0/val_npz/
data/LiTS/fold_0/test_vol_h5/
data/LiTS/fold_0/lists_LiTS/
```

The LiTS labels are converted to binary tumor masks: background/non-tumor is `0`, tumor is `1`.

## Train

Train one LiTS fold:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset LiTS \
  --root_path data/LiTS/fold_0/train_npz \
  --list_dir data/LiTS/fold_0/lists_LiTS \
  --vit_name R50-ViT-B_16 \
  --img_size 512 \
  --batch_size 4 \
  --output-dir output/lits_fold_0
```

Run preprocessing and training for folds `0..2`:

```bash
bash run.sh
```

Train Synapse:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16
```

Training writes logs and checkpoints to:

```text
<output-dir>/train.log
<output-dir>/checkpoints/log/
<output-dir>/checkpoints/epoch_<N>.pth
```

By default, `trainer.py` saves epoch checkpoints near the end of training. It does not currently create `best_model.pth` unless you add or copy one manually.

## Test

Test one LiTS fold:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset LiTS \
  --volume_path data/LiTS/fold_0/test_vol_h5 \
  --list_dir data/LiTS/fold_0/lists_LiTS \
  --vit_name R50-ViT-B_16 \
  --img_size 512 \
  --batch_size 4 \
  --output-dir output/lits_fold_0 \
  --checkpoint epoch_149.pth \
  --is_savenii
```

`--checkpoint` can be either a filename under `<output-dir>/checkpoints/` or a full checkpoint path.

Run the fold test script:

```bash
bash test.sh
```

Check `test.sh` before running: it currently uses `--output-dir outputs/lits_fold_<i>`, while `run.sh` trains into `output/lits_fold_<i>`. Use the same output directory for testing that you used for training, and choose an existing checkpoint filename such as `epoch_149.pth` unless `best_model.pth` exists.

Test Synapse:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16 \
  --output-dir outputs \
  --checkpoint path/to/checkpoint.pth
```

Test logs are written to `<output-dir>/test.log`. If `--is_savenii` is set, predictions are saved under:

```text
<output-dir>/predictions/<checkpoint_name>/
```

## Useful Files

- `tools/li_preprocess.py`: converts raw LiTS17 NIfTI files to TransUNet `.npz` slices and `.npy.h5` test volumes.
- `run.sh`: preprocesses and trains LiTS folds `0..2`.
- `test.sh`: tests LiTS folds `0..2`.
- `train.py`: training entry point.
- `test.py`: inference/evaluation entry point.

## References

- Original TransUNet paper: https://arxiv.org/pdf/2102.04306.pdf
- Google ViT: https://github.com/google-research/vision_transformer
- ViT-pytorch: https://github.com/jeonsworld/ViT-pytorch
