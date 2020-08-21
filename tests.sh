echo "====================SPIRAConvV2-MFCC===================="
# evaluation

echo "____________________evaluation____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/config.json  --batch_size 15 --num_workers 2


