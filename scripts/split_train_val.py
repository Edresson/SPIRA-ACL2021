import pandas as pd
import argparse
import re
import os 
from sklearn.model_selection import train_test_split

def options():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", help="Input CSV with Train data", required=True)
    args = parser.parse_args()
    return args

def main():
    # Get options
    args = options()
    # Open the file
    df_in = pd.read_csv(args.input).values[1:]
    x = []
    y = []
    for line in df_in:
        x.append(line)
        y.append(line[1])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    stratify=y)
    root_dir = os.path.split(args.input)[0]

    df = pd.DataFrame(x_train, columns=["file_path", "class", "sexo", "idade", "nivel_falta_de_ar"])
    df.to_csv(os.path.join(root_dir,'metadata_train.csv'), index=False)

    df = pd.DataFrame(x_test, columns=["file_path", "class", "sexo", "idade", "nivel_falta_de_ar"])
    df.to_csv(os.path.join(root_dir,'metadata_eval.csv'), index=False)

if __name__ == '__main__':
    main()