# SPIRA-Network

## Installing:

`pip install -r requirements.txt`

Alternatively, you can create an environment with the provided file.

`conda env create -f spira_environment.yml`
    
## Trainning the model:

```
python train.py [-c, --config_path] [--checkpoint_path]
    details:
        *'-c', '--config_path', type=str, required=True, help="json file with configurations get in checkpoint path"
        *'--checkpoint_path', type=str, default=None, required=True, help="path of checkpoint pt file, for continue training"
```

### Sample commands:

#### Training:

* (fresh start) python train.py -c ./config.json
* (from checkpoint) python train.py -c ./config.json --checkpoint_path ./checkpoint.pt

#### Testing:
* python test.py -t ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ -c ./config.json --checkpoint_path ./checkpoint.pt
        
## Testing the model:

```
python test.py [-t, --test_csv] [-r, --test_root_dir] [-c, --config_path] [--checkpoint_path] [--batch_size] [--num_workers] [--no_insert_noise]
    details:
        *'-t', '--test_csv', type=str, required=True, help="test csv example: ../SPIRA_Dataset_V1/metadata_test.csv"
        *'-r', '--test_root_dir', type=str, required=True, help="Test root dir example: ../SPIRA_Dataset_V1/"
        *'-c', '--config_path', type=str, required=True, help="json file with configurations get in checkpoint path"
        *'--checkpoint_path', type=str, default=None, required=True, help="path of checkpoint pt file, for continue training"
        *'--batch_size', type=int, default=20, help="Batch size for test"
        *'--num_workers', type=int, default=10, help="Number of Workers for test data load"
        *'--no_insert_noise', type=bool, default=False, help=" No insert noise in test ?"
```

### Analysing the model:

## Adding new models: 
   
1. Add option with model name in model_name property of config.json (or create another json configuration file);
2. Create a module in the models package and import your model class in train.py script;
3. Add an option for it in the if... else chain in test.py line 58
4. Import model from models in test.py. 
5. Add an option for it in the if... else chain in test.py line 77

