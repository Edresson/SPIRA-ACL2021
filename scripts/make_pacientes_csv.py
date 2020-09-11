import pandas as pd
import argparse
import re
import os 

def options():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", help="Input CSV file with all patients information. Exported from DB.", required=True)
    parser.add_argument("-d", "--dataset", help="audios patient  dir ex: pacientes/", required=True)
    parser.add_argument("-o", "--output_file", help="CSV file for output", required=False, default='metadata.csv')
    args = parser.parse_args()
    return args

def main():

    # Get options
    args = options()
    # Open the file
    df_in = pd.read_csv(args.input).values
    output_list = []
    valid_files_list = []
    lines = []
    files = os.listdir(args.dataset)
    for line in df_in[1:]:
        id_generate, telefone_origem, paciente_rghc, quando, sexo, idade, insp, oxigenacao, terapia, aceite_opus, palavras_opus, feedback = line 
        _, name = os.path.split(palavras_opus)
        if name in files:
            file_path = os.path.join(args.dataset,name)
            # file_path, class, sexo, idade, falta de ar(paciente sao setados como maxima falta de AR)
            lines.append([file_path,1, sexo, idade, 5])
    
    df = pd.DataFrame(lines, columns=["file_path", "class", "sexo", "idade", "nivel_falta_de_ar"])
    df.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()