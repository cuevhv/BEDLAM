import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
import torch
from utils_visualization import visualize
from utils_smplx import get_smplx_vertices
from utils_camera import get_cam_rotmat, get_cam_int, get_cam_trans, get_global_orient, get_bbox_valid, project

import cv2
import ipdb



def get_data(csv_data, cam_csv_base, gt_smplx_folder, image_folder_base, output_folder, fps, scene_name):
    scenes = []
    scene_data = {}

    # Parse csv file generated by Unreal for training
    for idx, comment in tqdm(enumerate(csv_data['Comment']), total=len(csv_data['Comment'])):
        if 'sequence_name' in comment:
            if len(scene_data) != 0:
                scenes.append(scene_data)
            scene_data = {}
            person_idx = 0
            #Get sequence name and corresponding camera details
            seq_name = comment.split(';')[0].split('=')[-1]
            cam_csv_data = pd.read_csv(os.path.join(cam_csv_base, seq_name+'_camera.csv'))
            cam_csv_data = cam_csv_data.to_dict('list')
            scene_data["cam_x"] = cam_csv_data['x']
            scene_data["cam_y"] = cam_csv_data['y']
            scene_data["cam_z"] = cam_csv_data['z']
            scene_data["cam_yaw_"] = cam_csv_data['yaw']
            scene_data["cam_pitch_"] = cam_csv_data['pitch']
            scene_data["cam_roll_"] = cam_csv_data['roll']
            scene_data["fl"] = cam_csv_data['focal_length']
            scene_data["fps"] = fps
            if 'closeup' in scene_name:
                scene_data["rotate_flag"] = True
                scene_data["sensor_w"] = 20.25
                scene_data["sensor_h"] = 36
                scene_data["img_w"] = 720
                scene_data["img_h"] = 1280
            else:
                scene_data["rotate_flag"] = False
                scene_data["sensor_w"] = 36
                scene_data["sensor_h"] = 20.25
                scene_data["img_w"] = 1280
                scene_data["img_h"]  = 720

            scene_data["bodies"] = []


        elif 'start_frame' in comment:
            # Get body details
            body_data = {}
            body_data["start_frame"] = int(comment.split(';')[0].split('=')[-1])
            body = csv_data['Body'][idx]
            person_id_ = body.split('_')
            body_data["person_id"] = '_'.join(person_id_[:-1])
            sequence_id = person_id_[-1]

            smplx_param_orig = np.load(os.path.join(gt_smplx_folder, body_data["person_id"], sequence_id, 'motion_seq.npz'))
            body_data["gender_sub"] = smplx_param_orig['gender_sub'].item()

            body_data["poses"] = smplx_param_orig['poses']
            body_data["trans"] = smplx_param_orig['trans']
            body_data["betas"] = smplx_param_orig['betas']
            body_data["motion_info"] = smplx_param_orig['motion_info']
            body_data["gender"] = smplx_param_orig['gender'].item()

            body_data["image_folder"] = os.path.join(image_folder_base, seq_name)
            body_data["trans_body"] = [csv_data['X'][idx], csv_data['Y'][idx], csv_data['Z'][idx]]
            body_data["body_yaw_"] = csv_data['Yaw'][idx]
            body_data["person_idx"] = str(person_idx).zfill(2)
            body_data["output_dir"] = output_folder
            body_data["fps"] = fps
            scene_data["bodies"].append(body_data)
            person_idx += 1
        else:
            continue

    return scenes


def process_scenes(scene_data, smplx_models, scale_factor, downsample_mat):

    all_images = sorted(glob(os.path.join(scene_data["bodies"][0]["image_folder"], '*')))

    for img_idx, image_path in (enumerate(all_images)):
        verts_bodies = []
        for body_data in scene_data["bodies"]:
            # Saving every 5th frame
            if scene_data["fps"] == 6:
                if img_idx % 5 != 0:
                    continue
                smplx_param_ind = img_idx*5+body_data["start_frame"]
                cam_ind = img_idx
            else:
                smplx_param_ind = img_idx+body_data["start_frame"]
                cam_ind = img_idx

            if smplx_param_ind > body_data['poses'].shape[0]:
                break
            pose = body_data['poses'][smplx_param_ind]
            transl = body_data['trans'][smplx_param_ind]
            beta = body_data['betas']
            motion_info = body_data['motion_info']

            gender = body_data['gender']
            cam_pitch_ind = -scene_data["cam_pitch_"][cam_ind]
            cam_yaw_ind = -scene_data["cam_yaw_"][cam_ind]

            if scene_data["rotate_flag"]:
                cam_roll_ind = -scene_data["cam_roll_"][cam_ind] + 90
            else:
                cam_roll_ind = -scene_data["cam_roll_"][cam_ind]

            cam_int = get_cam_int(scene_data["fl"][cam_ind], scene_data["sensor_w"], scene_data["sensor_h"],
                                  scene_data["img_w"]/2., scene_data["img_h"]/2.)

            body_rotmat, cam_rotmat_for_trans = get_cam_rotmat(body_data["body_yaw_"], cam_pitch_ind, cam_yaw_ind, cam_roll_ind)
            cam_t = [scene_data["cam_x"][cam_ind], scene_data["cam_y"][cam_ind], scene_data["cam_z"][cam_ind]]
            cam_trans = get_cam_trans(body_data["trans_body"], cam_t)
            cam_trans = np.matmul(cam_rotmat_for_trans, cam_trans.T).T

            w_global_orient, c_global_orient, c_trans, w_trans, cam_rotmat = get_global_orient(pose, beta, transl, gender,
                                                                                               body_data["body_yaw_"], cam_pitch_ind,
                                                                                               cam_yaw_ind, cam_roll_ind, cam_trans,
                                                                                               smplx_models)
            cam_ext_ = np.zeros((4, 4))
            cam_ext_[:3, :3] = cam_rotmat
            cam_ext_trans = np.concatenate([cam_trans, np.array([[1]])],axis=1)
            cam_ext_[:, 3] = cam_ext_trans

            pose_cam = pose.copy()
            pose_cam[:3] = c_global_orient

            pose_world = pose.copy()
            pose_world[:3] = w_global_orient

            vertices3d, joints3d = get_smplx_vertices(pose_cam, beta, c_trans, gender, smplx_models)
            joints2d = project(joints3d, torch.tensor(cam_trans), cam_int)
            vertices3d_downsample = downsample_mat.matmul(vertices3d)

            proj_verts = project(vertices3d_downsample, torch.tensor(cam_trans), cam_int)

            center, scale, num_vis_joints, bbox = get_bbox_valid(joints2d[:22], rescale=scale_factor,
                                                                 img_width=scene_data["img_w"], img_height=scene_data["img_h"])
            if center[0] < 0 or center[1] < 0 or scale <= 0:
                continue

            #visualize_crop(image_path, center, scale, torch.tensor(verts_cam2) , cam_int[0][0], smplx_model_male.faces)
            if num_vis_joints < 12:
                continue

            verts_cam2 = vertices3d.detach().cpu().numpy() + cam_trans
            verts_bodies.append(verts_cam2)


            out_img_dir = os.path.join(body_data["output_dir"], image_path.split('/')[-4])
            os.makedirs(out_img_dir, exist_ok=True)
            seq_fn = os.path.normpath(image_path).split(os.sep)[-2]

            visualize(image_path, verts_cam2[None, ], cam_int[0][0],
                      smplx_models["male"].faces, body_data["person_idx"],
                      out_img_dir, scene_data["rotate_flag"])
            # visualize_2d(image_path, joints2d)

            npz_name = os.path.basename(image_path).replace(".png", f"_{body_data['person_idx']}.npz")
            np.savez(
                os.path.join(out_img_dir, npz_name),
                imgname=os.path.join(seq_fn, os.path.basename(image_path)),
                center=center,
                scale=scale,
                pose_cam=pose_cam,
                pose_world=pose_world,
                shape=beta,
                trans_cam=c_trans,
                trans_world=w_trans,
                cam_int=cam_int,
                cam_ext=cam_ext_,
                gender=body_data["gender_sub"],
                vertices2d=proj_verts,
                joints2d=joints2d,
                vertices3d=verts_cam2, #vertices3d.detach().cpu().numpy(),
                joints3d=joints3d.detach().cpu().numpy(),
                motion_info=motion_info,
                sub=body_data["person_id"],
                person_idx=body_data["person_idx"],
            )

        verts_bodies = np.array(verts_bodies)

        if verts_bodies.size:
            visualize(image_path, verts_bodies, cam_int[0][0],
                        smplx_models["male"].faces, "all",
                        out_img_dir, scene_data["rotate_flag"])


def parallel_process_scenes(args):
    scene_data, smplx_models, scale_factor, downsample_mat = args
    print(scene_data["bodies"][0]["image_folder"])

    all_images = sorted(glob(os.path.join(scene_data["bodies"][0]["image_folder"], '*')))

    for img_idx, image_path in (enumerate(all_images)):
        verts_bodies = []
        for body_data in scene_data["bodies"]:
            # Saving every 5th frame
            if scene_data["fps"] == 6:
                if img_idx % 5 != 0:
                    continue
                smplx_param_ind = img_idx*5+body_data["start_frame"]
                cam_ind = img_idx
            else:
                smplx_param_ind = img_idx+body_data["start_frame"]
                cam_ind = img_idx

            if smplx_param_ind > body_data['poses'].shape[0]:
                break
            pose = body_data['poses'][smplx_param_ind]
            transl = body_data['trans'][smplx_param_ind]
            beta = body_data['betas']
            motion_info = body_data['motion_info']

            gender = body_data['gender']
            cam_pitch_ind = -scene_data["cam_pitch_"][cam_ind]
            cam_yaw_ind = -scene_data["cam_yaw_"][cam_ind]

            if scene_data["rotate_flag"]:
                cam_roll_ind = -scene_data["cam_roll_"][cam_ind] + 90
            else:
                cam_roll_ind = -scene_data["cam_roll_"][cam_ind]

            cam_int = get_cam_int(scene_data["fl"][cam_ind], scene_data["sensor_w"], scene_data["sensor_h"],
                                  scene_data["img_w"]/2., scene_data["img_h"]/2.)

            body_rotmat, cam_rotmat_for_trans = get_cam_rotmat(body_data["body_yaw_"], cam_pitch_ind, cam_yaw_ind, cam_roll_ind)
            cam_t = [scene_data["cam_x"][cam_ind], scene_data["cam_y"][cam_ind], scene_data["cam_z"][cam_ind]]
            cam_trans = get_cam_trans(body_data["trans_body"], cam_t)
            cam_trans = np.matmul(cam_rotmat_for_trans, cam_trans.T).T

            w_global_orient, c_global_orient, c_trans, w_trans, cam_rotmat = get_global_orient(pose, beta, transl, gender,
                                                                                               body_data["body_yaw_"], cam_pitch_ind,
                                                                                               cam_yaw_ind, cam_roll_ind, cam_trans,
                                                                                               smplx_models)
            cam_ext_ = np.zeros((4, 4))
            cam_ext_[:3, :3] = cam_rotmat
            cam_ext_trans = np.concatenate([cam_trans, np.array([[1]])],axis=1)
            cam_ext_[:, 3] = cam_ext_trans

            pose_cam = pose.copy()
            pose_cam[:3] = c_global_orient

            pose_world = pose.copy()
            pose_world[:3] = w_global_orient

            vertices3d, joints3d = get_smplx_vertices(pose_cam, beta, c_trans, gender, smplx_models)
            joints2d = project(joints3d, torch.tensor(cam_trans), cam_int)
            vertices3d_downsample = downsample_mat.matmul(vertices3d)

            proj_verts = project(vertices3d_downsample, torch.tensor(cam_trans), cam_int)

            center, scale, num_vis_joints, bbox = get_bbox_valid(joints2d[:22], rescale=scale_factor,
                                                                 img_width=scene_data["img_w"], img_height=scene_data["img_h"])
            if center[0] < 0 or center[1] < 0 or scale <= 0:
                continue

            #visualize_crop(image_path, center, scale, torch.tensor(verts_cam2) , cam_int[0][0], smplx_model_male.faces)
            if num_vis_joints < 12:
                continue

            verts_cam2 = vertices3d.detach().cpu().numpy() + cam_trans
            verts_bodies.append(verts_cam2)


            out_img_dir = os.path.join(body_data["output_dir"], image_path.split('/')[-4])
            os.makedirs(out_img_dir, exist_ok=True)
            seq_fn = os.path.normpath(image_path).split(os.sep)[-2]

            visualize(image_path, verts_cam2[None, ], cam_int[0][0],
                      smplx_models["male"].faces, body_data["person_idx"],
                      out_img_dir, scene_data["rotate_flag"])
            # visualize_2d(image_path, joints2d)

            npz_name = os.path.basename(image_path).replace(".png", f"_{body_data['person_idx']}.npz")
            np.savez(
                os.path.join(out_img_dir, npz_name),
                imgname=os.path.join(seq_fn, os.path.basename(image_path)),
                center=center,
                scale=scale,
                pose_cam=pose_cam,
                pose_world=pose_world,
                shape=beta,
                trans_cam=c_trans,
                trans_world=w_trans,
                cam_int=cam_int,
                cam_ext=cam_ext_,
                gender=body_data["gender_sub"],
                vertices2d=proj_verts,
                joints2d=joints2d,
                vertices3d=verts_cam2, #vertices3d.detach().cpu().numpy(),
                joints3d=joints3d.detach().cpu().numpy(),
                motion_info=motion_info,
                sub=body_data["person_id"],
                person_idx=body_data["person_idx"],
                )

        verts_bodies = np.array(verts_bodies)

        if verts_bodies.size:
            visualize(image_path, verts_bodies, cam_int[0][0],
                        smplx_models["male"].faces, "all",
                        out_img_dir, scene_data["rotate_flag"])