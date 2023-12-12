import numpy as np
import cv2
import os
import torch
from renderer_pyrd_nearest_n import Renderer


def visualize(image_path, verts, focal_length, smpl_faces, person_idx,
              output_dir, rotate_flag=False, hbmi_texture=None,
              render_background=False, render_labels=True, save_background=False):
    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w, c = img.shape

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                        faces=smpl_faces, render_labels=render_labels)
    front_view = renderer.render_front_view(verts,
                                            bg_img_rgb = img[:, :, ::-1].copy() if render_background else None,
                                            hbmi_texture=hbmi_texture)

    img_name = os.path.basename(image_path).replace(".png", f"_{person_idx}.png")
    img_name = os.path.basename(image_path).replace(".jpg", f"_{person_idx}.png")
    cv2.imwrite(os.path.join(output_dir, img_name), front_view[:, :, ::-1])

    if save_background:
        background = ((front_view.mean(axis=-1) != 0) * 255).astype(np.uint8)
        background_output_dir = output_dir.replace("body_correspondence", "mask")
        os.makedirs(background_output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(background_output_dir, img_name), background)

