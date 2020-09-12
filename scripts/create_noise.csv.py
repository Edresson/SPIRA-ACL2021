import os
import argparse
import re

if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", required=True)
    args = parser.parse_args()

    list_dir = os.listdir(args.path)
    with open('noise_data.csv', 'w') as the_file:
        for f in list_dir:
            full_path = os.path.join(args.path, f)
            _, ext = os.path.splitext(f)
            if not '.wav' in f:
                wav_name = re.sub(r'\s+', '-',full_path.replace(ext,'.wav').replace(' ',''))
                print(full_path,wav_name)
                os.system('ffmpeg -y -i "'+full_path+'" "'+wav_name+'"')
                full_path = wav_name
            else:
                continue
            the_file.write(full_path+'\n')


