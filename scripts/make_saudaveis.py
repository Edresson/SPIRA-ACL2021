import pandas as pd
import argparse
import re
import os 
import json


def options():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-j", "--json_path", help=" Path from Control json directory", required=True)
    parser.add_argument("-d", "--dataset", help="audios control dir ex: control/", required=True)
    parser.add_argument("-o", "--output_file", help="CSV file for output", required=False, default='metadata.csv')
    args = parser.parse_args()
    return args

def main():

    # Get options
    args = options()
    output_list = []
    valid_files_list = []
    lines = []
    files = os.listdir(args.dataset)
    json_dir = args.json_path
    for name in files:
        # considerar apenas pessoas apartir dos 40 anos
        try:
            #print(os.path.join(json_dir,file.replace('.wav','.json')))
            file_name = name.split('_')[0]+'.json'
            #print('aquii',int(json.load(open(os.path.join(json_dir,file_name)))['idade']))
            json_file = json.load(open(os.path.join(json_dir,file_name)))
            print(int(json_file["faltaDeAr"]))
            # int(json['idade']) < 99 or
            #if not int(json_file["faltaDeAr"]) != 0:
            #    continue
        except Exception as e:
            print(name, 'ignorado pelo erro :', e)
            continue
        file_path = os.path.join(args.dataset,name)
        # file_path, class, sexo, idade, falta de ar(paciente sao setados como maxima falta de AR)
        idade = int(json_file['idade'])
        sexo = json_file["genero"]
        if sexo == "Masculino":
            sexo = 'M'
        elif sexo == "Feminino":
            sexo = "F"
        lines.append([file_path, 0, sexo, idade, int(json_file["faltaDeAr"])])
        
    df = pd.DataFrame(lines, columns=["file_path", "class", "sexo", "idade", "nivel_falta_de_ar"])
    df.to_csv(args.output_file, mode='a', index=False, header=False)

if __name__ == '__main__':
    main()