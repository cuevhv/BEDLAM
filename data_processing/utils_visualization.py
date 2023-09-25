import numpy as np
import cv2
import os
import torch
from renderer_pyrd_nearest_n import Renderer


def visualize(image_path, verts, focal_length, smpl_faces, person_idx,
              output_dir, rotate_flag=False):
    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w, c = img.shape

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                        faces=smpl_faces)
    front_view = renderer.render_front_view(verts,
                                            # bg_img_rgb=img[:, :, ::-1].copy()
                                            )

    img_name = os.path.basename(image_path).replace(".png", f"_{person_idx}.png")
    cv2.imwrite(os.path.join(output_dir, img_name), front_view[:, :, ::-1])