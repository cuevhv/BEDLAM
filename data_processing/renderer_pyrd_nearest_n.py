# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import trimesh
import numpy as np
import colorsys
import time
import cv2
from trimesh.visual.texture import SimpleMaterial
from trimesh.visual.texture import TextureVisuals
from PIL import Image
import os

# Disable antialiasing:
import remove_antialiasing
import pyrender

def fix_mesh_shape(mesh, vt, f, ft):
    '''
    Add missing vertices to the mesh such that it has the same number of vertices as the texture coordinates
    mesh: 3D vertices of the orginal mesh
    vt: 2D vertices of the texture map
    f: 3D faces of the orginal mesh (0-indexed)
    ft: 2D faces of the texture map (0-indexed)
    '''

    #build a correspondance dictionary from the original mesh indices to the (possibly multiple) texture map indices
    f_flat = f.flatten()
    ft_flat = ft.flatten()
    correspondances = {}

    #traverse and find the corresponding indices in f and ft
    for i in range(len(f_flat)):
        if f_flat[i] not in correspondances:
            correspondances[f_flat[i]] = [ft_flat[i]]
        else:
            if ft_flat[i] not in correspondances[f_flat[i]]:
                correspondances[f_flat[i]].append(ft_flat[i])

    #build a mesh using the texture map vertices
    new_mesh = np.zeros((vt.shape[0], 3))
    for old_index, new_indices in correspondances.items():
        for new_index in new_indices:
            new_mesh[new_index] = mesh[old_index]

    #define new faces using the texture map faces
    f_new = ft

    return new_mesh, f_new


def vertex_to_face_color(vertex_colors, faces, nearest_neighbor=False):
    """
    Convert a list of vertex colors to face colors.

    Parameters
    ----------
    vertex_colors: (n,(3,4)),  colors
    faces:         (m,3) int, face indexes

    Returns
    -----------
    face_colors: (m,4) colors
    """
    if nearest_neighbor:
        face_colors = vertex_colors[faces[:, 0]]
    else:
        face_colors = vertex_colors[faces].mean(axis=1)
    return face_colors.astype(np.uint8)


class Renderer(object):

    def __init__(self, focal_length=600, img_w=512, img_h=512, faces=None,
                 same_mesh_color=False):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0,)
        self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 1)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)

        # Create a material object from the texture image
        texture_image = Image.open("bedlam_data/body_models/smplx/models/smplx/smpl_colorcoded_test.png").convert('RGB')
        material = SimpleMaterial(image=texture_image, diffuse=[1.0, 1.0, 1.0, 1.0])

        time_s = time.time()
        with open("bedlam_data/body_models/smplx/models/smplx/smplx_uv_meshcapade.obj", "r") as f:
            x = trimesh.exchange.obj.load_obj(f)
            # print(x.keys())

        # for every person in the scene
        for n in range(num_people):

            # Create a texture visuals object
            uv = x["visual"].uv
            uv2d = x["faces"]

            time_s = time.time()
            new_verts, new_faces = fix_mesh_shape(verts[n], uv, self.faces, uv2d)

            mesh = trimesh.Trimesh(new_verts, new_faces, process=False,)

            apply_texture = True
            if apply_texture:
                time_s = time.time()
                texture_visuals = TextureVisuals(uv=uv, image=texture_image, material=material)
                colors = texture_visuals.material.to_color(uv)
                # print(np.unique(colors, return_index=True))
                face_colors = vertex_to_face_color(colors, new_faces, nearest_neighbor=True)
                face_colors[..., 1] = n+1
                mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, face_colors=face_colors)

            else:

                if self.same_mesh_color:
                    mesh_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
                else:
                    mesh_color = colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=mesh_color)
                mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, vertex_colors=mesh_color)
                mesh.vertices = new_verts
                mesh.faces = new_faces

            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, wireframe=False, smooth=False)
            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        scene.sigma = 0
        time_s = time.time()
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
        self.renderer.delete()

        color_rgb = color_rgba[:, :, :3]
        if bg_img_rgb is None:
            return color_rgb
        else:
            mask = depth_map > 0
            bg_img_rgb[mask] = color_rgb[mask]
            return bg_img_rgb

    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()
