

# SPIRAConvV1
echo "====================SPIRAConvV1-MEL_SPEC===================="
# evaluation

echo "____________________evaluation____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MEL_SPEC/spiraconv_v1_75/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MEL_SPEC/spiraconv_v1_75/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MEL_SPEC/spiraconv_v1_75/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MEL_SPEC/spiraconv_v1_75/config.json  --batch_size 15 --num_workers 2
echo "____________________test____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MEL_SPEC/spiraconv_v1_75/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MEL_SPEC/spiraconv_v1_75/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MEL_SPEC/spiraconv_v1_75/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MEL_SPEC/spiraconv_v1_75/config.json  --batch_size 15 --num_workers 2



echo "====================SPIRAConvV1-MFCC===================="
# evaluation

echo "____________________evaluation____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_75/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_75/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_75/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_75/config.json  --batch_size 15 --num_workers 2
echo "____________________test____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_75/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_75/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_75/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_75/config.json  --batch_size 15 --num_workers 2


echo "====================SPIRAConvV2-MFCC===================="
# evaluation

echo "____________________evaluation____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/config.json  --batch_size 15 --num_workers 2
echo "____________________test____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v2_78/config.json  --batch_size 15 --num_workers 2



# SPIRAConvV1

echo "====================SPIRAConvV1-MFCC-weightdecay===================="
# evaluation

echo "____________________evaluation____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_73/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_73/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_eval.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_73/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_73/config.json  --batch_size 15 --num_workers 2
echo "____________________test____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_73/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_73/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_73/best_checkpoint.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1_MFCC/spiraconv_v1_73/config.json  --batch_size 15 --num_workers 2
