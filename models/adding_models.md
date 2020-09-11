# Spira-Network

## Adding new model: 
   
1. Add option with model name in model_name property of config.json (or create another json configuration file);
2. Create a module in the models folder and import your model class in train.py script;
3. Add an option for it in the if... else chain in test.py line 58
4. Import model from models in test.py. 
5. Add an option for it in the if... else chain in test.py line 77
