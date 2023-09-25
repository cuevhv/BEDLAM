from utils_smplx import get_smplx_vertices
import numpy as np
import torch
import cv2


def get_global_orient(pose, beta, transl, gender, body_yaw, cam_pitch, cam_yaw, cam_roll, cam_trans, smplx_models):
    # World coordinate transformation after assuming camera has 0 yaw and is at origin
    body_rotmat, _ = cv2.Rodrigues(np.array([[0, ((body_yaw - 90+cam_yaw) / 180) * np.pi, 0]], dtype=float))
    pitch_rotmat, _ = cv2.Rodrigues(np.array([cam_pitch / 180 * np.pi, 0, 0]).reshape(3, 1))
    roll_rotmat, _ = cv2.Rodrigues(np.array([0., 0, cam_roll / 180 * np.pi, ]).reshape(3, 1))
    final_rotmat = np.matmul(roll_rotmat, (pitch_rotmat))

    transform_coordinate = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    transform_body_rotmat = np.matmul(body_rotmat, transform_coordinate)
    w_global_orient = cv2.Rodrigues(np.dot(transform_body_rotmat, cv2.Rodrigues(pose[:3])[0]))[0].T[0]

    #apply rotation transformation to translation
    verts_local, joints_local = get_smplx_vertices(pose, beta, torch.zeros(3), gender, smplx_models)
    j0 = joints_local[0].detach().cpu().numpy()
    rot_j0 = np.matmul(transform_body_rotmat, j0.T).T
    l_translation_ = np.matmul(transform_body_rotmat, transl.T).T
    l_translation = rot_j0 + l_translation_
    w_translation = l_translation - j0

    c_global_orient = cv2.Rodrigues(np.dot(final_rotmat, cv2.Rodrigues(w_global_orient)[0]))[0].T[0]
    c_translation = np.matmul(final_rotmat, l_translation.T).T - j0

    return w_global_orient, c_global_orient, c_translation, w_translation, final_rotmat



def get_bbox_valid(joints, img_height, img_width, rescale):
    #Get bbox using keypoints
    valid_j = []
    joints = np.copy(joints)
    for j in joints:
        if j[0] > img_width or j[1] > img_height or j[0] < 0 or j[1] < 0:
            continue
        else:
            valid_j.append(j)

    if len(valid_j) < 1:
        return [-1, -1], -1, len(valid_j), [-1, -1, -1, -1]

    joints = np.array(valid_j)

    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]

    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

    scale *= rescale
    return center, scale, len(valid_j), bbox


def get_cam_int(fl, sens_w, sens_h, cx, cy):
    flx = focalLength_mm2px(fl, sens_w, cx)
    fly = focalLength_mm2px(fl, sens_h, cy)

    cam_mat = np.array([[flx, 0, cx],
                       [0, fly, cy],
                       [0, 0, 1]])
    return cam_mat


def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def unreal2cv2(points):
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1.0, -1.0, 1.0])
    return points


def project(points, cam_trans, cam_int):
    points = points + cam_trans
    cam_int = torch.tensor(cam_int).float()

    projected_points = points / points[:, -1].unsqueeze(-1)
    projected_points = torch.einsum('ij, kj->ki', cam_int, projected_points.float())

    return projected_points.detach().cpu().numpy()


def get_cam_trans(body_trans, cam_trans):
    cam_trans = np.array(cam_trans) / 100
    cam_trans = unreal2cv2(np.reshape(cam_trans, (1, 3)))

    body_trans = np.array(body_trans) / 100
    body_trans = unreal2cv2(np.reshape(body_trans, (1, 3)))

    trans = body_trans - cam_trans
    return trans


def get_cam_rotmat(body_yaw, pitch, yaw, roll):
    #Because bodies are rotation by 90
    body_rotmat, _ = cv2.Rodrigues(np.array([[0, ((body_yaw - 90) / 180) * np.pi, 0]], dtype=float))
    rotmat_yaw, _ = cv2.Rodrigues(np.array([[0, ((yaw) / 180) * np.pi, 0]], dtype=float))
    rotmat_pitch, _ = cv2.Rodrigues(np.array([pitch / 180 * np.pi, 0, 0]).reshape(3, 1))
    rotmat_roll, _ = cv2.Rodrigues(np.array([0, 0, roll / 180 * np.pi]).reshape(3, 1))
    final_rotmat = np.matmul(rotmat_roll, np.matmul(rotmat_pitch, rotmat_yaw))
    return body_rotmat, final_rotmat