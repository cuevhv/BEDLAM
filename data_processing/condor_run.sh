export PYTHONBUFFERED=1
export PATH=$PATH
/home/hcuevas/miniconda3/envs/gen_bedlam/bin/python df_full_body_parallel.py --img_folder $1 --output_folder $2 --bedlam_scene_csv $3 --fps 1