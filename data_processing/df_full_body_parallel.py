""" python df_full_body_parallel.py --img_folder /media/hcuevas/7E602496602456E5/temp \
    --output_folder /media/hcuevas/7E602496602456E5/temp/temp/
"""
import os
import time
import pickle
import csv
import argparse
import pandas as pd
from glob import glob
from multiprocessing.pool import Pool
from tqdm import tqdm
from utils_load_data import get_data, process_scenes, parallel_process_scenes, parallel_process_scenes2
from utils_smplx import get_smplx_models
import concurrent.futures
import ipdb

CLIFF_SCALE_FACTOR_BBOX = 1.2
MODEL_FOLDER = 'bedlam_data/body_models/smplx/models/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default='bedlam_data/images')
    parser.add_argument('--output_folder', type=str, default='bedlam_data/processed_labels')
    parser.add_argument('--smplx_gt_folder', type=str, default='bedlam_data/smplx_gt/neutral_ground_truth_motioninfo')
    parser.add_argument('--fps', type=int, default=6, help='6/30 fps output. With 6fps then every 5th frame is stored')

    args = parser.parse_args()
    base_image_folder = args.img_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gt_smplx_folder = args.smplx_gt_folder
    fps = args.fps

    image_folders = csv.reader(open('bedlam_scene_test.csv', 'r')) # File to parse folders
    next(image_folders)  # Skip header
    image_dict, npz_dict = {}, {}

    downsample_mat = pickle.load(open('downsample_mat_smplx.pkl', 'rb'))
    smplx_models = get_smplx_models(MODEL_FOLDER)

    for row in image_folders:
        image_dict[row[1]] = os.path.join(base_image_folder, row[0],'png')
        npz_dict[row[1]] = os.path.join(output_folder, str(row[0])+'.npz')

    for scene_name, v in tqdm(image_dict.items()):
        image_folder_base = v
        base_folder = v.replace('/png','')
        outfile = npz_dict[scene_name]
        csv_path = os.path.join(base_folder, 'be_seq.csv')
        csv_data = pd.read_csv(csv_path)
        csv_data = csv_data.to_dict('list')
        cam_csv_base = os.path.join(base_folder, 'ground_truth/camera')
        scenes = get_data(csv_data, cam_csv_base, gt_smplx_folder, image_folder_base, output_folder, fps, scene_name)


    smplx_models_list = [smplx_models for i in range(len(scenes))]
    scale_factor = [CLIFF_SCALE_FACTOR_BBOX for i in range(len(scenes))]
    downsample_mat_list = [downsample_mat for i in range(len(scenes))]
    args_parallel = list(zip(scenes, smplx_models_list, scale_factor, downsample_mat_list))

    parallel = True

    s_time = time.time()
    if parallel:
        cpus = 6  # os.cpu_count()
        print("total number of cpus: ", cpus)
        with concurrent.futures.ProcessPoolExecutor(cpus) as pool:
            pool.map(parallel_process_scenes2, args_parallel)
    else:
        for scene in scenes:
            process_scenes(scene, smplx_models, CLIFF_SCALE_FACTOR_BBOX, downsample_mat)

    print(f"total time taken: {time.time() - s_time}, for a total of {len(scenes)}")