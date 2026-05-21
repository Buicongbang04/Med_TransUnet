echo "#########################################################"
echo "START RUNNING CROSS-VALIDATION FOR LITS17 USING TRANSUNET"
echo "#########################################################"
echo ""
echo ""

SPLIT_PATH="data/splits"

for i in {0..2}
do
    echo "========================================================================="
    echo "Processing fold ${i}..."
    echo "========================================================================="
    echo ""

    echo "Creating new folder ${i}..."
    mkdir -p data/LiTS/fold_${i}
    echo ""

    # =========================================================
    # Preprocess
    # =========================================================
    echo "========================================================================="
    echo "Running preprocess script for fold ${i}..."
    echo "========================================================================="

    python tools/li_preprocess.py \
        --lits-root data/LiTS17 \
        --out-root data/LiTS/fold_${i} \
        --img-size 512 \
        --split-file ${SPLIT_PATH}/liver_fold_${i}.json \
        --clean

    if [ $? -ne 0 ]; then
        echo "Preprocess failed for fold ${i}"
        exit 1
    fi

    echo "========================================================================="
    echo "Finished preprocess for fold ${i}"
    echo "========================================================================="
    echo ""
    echo ""

    # =========================================================
    # Training
    # =========================================================
    echo "========================================================================="
    echo "Running training script for fold ${i}..."
    echo "========================================================================="

    CUDA_VISIBLE_DEVICES=0 python train.py \
        --dataset LiTS \
        --root_path data/LiTS/fold_${i}/train_npz \
        --list_dir data/LiTS/fold_${i}/lists_LiTS \
        --vit_name R50-ViT-B_16 \
        --img_size 512 \
        --batch_size 4 \
        --output-dir output/lits_fold_${i}

    if [ $? -ne 0 ]; then
        echo "Training failed for fold ${i}"
        exit 1
    fi

    echo "========================================================================="
    echo "Finished training for fold ${i}"
    echo "========================================================================="
    echo ""
    echo ""
done

echo "========================================================================="
echo "ALL FOLDS COMPLETED"
echo "========================================================================="