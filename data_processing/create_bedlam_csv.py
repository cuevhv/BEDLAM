import csv
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_list', type=str, required=True)
    parser.add_argument('--out_folder', type=str, required=True)
    args = parser.parse_args()

    with open(args.csv_list, 'r') as f:
        csv_list = f.readlines()
        header = csv_list[0].strip()
        for csv_file in csv_list[1:]:
            file_name = csv_file.strip().split(',')[0] + ".csv"
            file_name = os.path.join(args.out_folder, file_name)

            with open(file_name, 'w') as output_file:
                output_file.write(header + '\n')
                output_file.write(csv_file.strip() + '\n')

if __name__ == '__main__':
    main()