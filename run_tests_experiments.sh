
<< 'MULTILINE-COMMENT'
echo "==================== Experiment 1.1 ===================="
# evaluation

echo "____________________EVALUATION____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.1/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.1/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True 
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.1/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.1/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0
echo "____________________TESTE____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.1/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.1/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.1/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.1/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0

echo "========================================================="

echo "==================== Experiment 1.2 ===================="
# evaluation

echo "____________________EVALUATION____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.2/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.2/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True 
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.2/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.2/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0
echo "____________________TESTE____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.2/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.2/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.2/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.2/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0

echo "========================================================="

echo "==================== Experiment 1.3 ===================="
# evaluation

echo "____________________EVALUATION____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.3/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.3/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True 
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.3/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.3/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0
echo "____________________TESTE____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.3/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.3/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp1.3/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp1.3/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0

echo "========================================================="

echo "==================== Experiment 2.1 ===================="
# evaluation

echo "____________________EVALUATION____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.1/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.1/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True 
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.1/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.1/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0
echo "____________________TESTE____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.1/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.1/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.1/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.1/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0

echo "========================================================="


echo "==================== Experiment 2.2 ===================="
# evaluation

echo "____________________EVALUATION____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.2/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.2/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True 
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.2/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.2/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0
echo "____________________TESTE____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.2/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.2/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.2/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.2/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0

echo "========================================================="

echo "==================== Experiment 2.3 ===================="
# evaluation

echo "____________________EVALUATION____________________"
# without noise
echo "without noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.3/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.3/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True 
# with noise
echo "with noise:"
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_eval.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.3/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.3/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0
echo "____________________TESTE____________________"
# test
# without noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.3/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.3/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --no_insert_noise True
# with noise
python test.py --test_csv ../SPIRA_Dataset_V2/metadata_test.csv -r ../SPIRA_Dataset_V2/ --checkpoint_path ../checkpoints/Paper-Experiments/exp2.3/spiraconv_v2/best_checkpoint.pt --config_path ../checkpoints/Paper-Experiments/exp2.3/spiraconv_v2/config.json  --batch_size 15 --num_workers 2 --num_noise_control 1 --num_noise_patient 0

echo "========================================================="


MULTILINE-COMMENT