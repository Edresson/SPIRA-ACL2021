# SPIRAConvV1
echo "====================SPIRAConvV1===================="
# evaluation

echo "____________________evaluation____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1/config.json  --batch_size 15 --num_workers 2
echo "____________________test____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1/config.json  --batch_size 15 --num_workers 2

