for i in 0
do 
    echo "###########################"
    echo "Starting Test for fold ${i}..."
    echo "###########################"
    echo ""

    echo "-----------------"
    echo "Testing fold ${i}..."
    echo "-----------------"
    python test.py \
        --dataset LiTS \
        --volume_path data/LiTS/fold_${i}/test_vol_h5 \
        --list_dir data/LiTS/fold_${i}/lists_LiTS \
        --vit_name R50-ViT-B_16 \
        --img_size 512 \
        --batch_size 4 \
        --output-dir output/lits_fold_${i} \
        --checkpoint output/lits_fold_${i}/checkpoints/best_model.pth \
        --is_savenii
    if [ $? -ne 0 ]; then
        echo "Testing failed for fold ${i}"
        exit 1
    fi
    echo "Finished testing for fold ${i}"
    echo "-----------------"
    echo ""
    echo ""
done
