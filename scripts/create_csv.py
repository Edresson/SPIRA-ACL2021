import os
import argparse


if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", required=True)
    parser.add_argument('-c', '--class_id', type=int, required=True,
                        help='class 0 or 1')
    args = parser.parse_args()

    list_dir = os.listdir(args.path)
    with open('data.csv', 'a') as the_file:
        for f in list_dir:
            full_path = os.path.join(args.path, f)
            _, ext = os.path.splitext(f)
            if not '.wav' in f:
                wav_name = full_path.replace(ext,'.wav')
                os.system('ffmpeg -y -i '+full_path+' '+wav_name)
                full_path = wav_name
            
            the_file.write(full_path+','+str(args.class_id)+'\n')


