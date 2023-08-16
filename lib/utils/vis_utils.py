import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from datetime import datetime
import numpy as np
import torch
import math
import cv2
import trimesh
import pyrender
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from core.config import cfg
from core.logger import logger

from dataset.multiple_datasets import MultipleDatasets
from funcs_utils import rot6d_to_axis_angle
from human_models import smpl


def vis_bboxes(img, boxes):
    img = img.copy()
    color, thickness = (0, 255, 0), 2

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[0])+int(box[2]), int(box[1])+int(box[3])
    
        pos1 = (x_min, y_min)
        pos2 = (x_min, y_max)
        pos3 = (x_max, y_min)
        pos4 = (x_max, y_max)
        
        img = cv2.line(img, pos1, pos2, color, thickness) 
        img = cv2.line(img, pos1, pos3, color, thickness) 
        img = cv2.line(img, pos2, pos4, color, thickness) 
        img = cv2.line(img, pos3, pos4, color, thickness) 

    return img


def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((len(kps),1))], axis=1)

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    kp_mask = np.ascontiguousarray(kp_mask, dtype=np.uint8)

    # Draw the keypoints.
    for i in range(len(kps)):
        if kps[i][2] > 0:
            p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
            cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
            
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints_with_skeleton(img, kps, kps_line, bbox=None, kp_thre=0.4, alpha=1):
    # Convert form plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_line))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    
    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((len(kps),1))], axis=1)
    
    # Perfrom the drawing on a copy of the image, to allow for blending
    kp_mask = np.copy(img)

    # Draw bounding box
    if bbox is not None:
        b1 = bbox[0, 0].astype(np.int32), bbox[0, 1].astype(np.int32)
        b2 = bbox[1, 0].astype(np.int32), bbox[1, 1].astype(np.int32)
        b3 = bbox[2, 0].astype(np.int32), bbox[2, 1].astype(np.int32)
        b4 = bbox[3, 0].astype(np.int32), bbox[3, 1].astype(np.int32)

        cv2.line(kp_mask, b1, b2, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b2, b3, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b3, b4, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b4, b1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    # Draw the keypoints
    for l in range(len(kps_line)):
        i1 = kps_line[l][0]
        i2 = kps_line[l][1]
        
        p1 = kps[i1,0].astype(np.int32), kps[i1,1].astype(np.int32)
        p2 = kps[i2,0].astype(np.int32), kps[i2,1].astype(np.int32)
        if kps[i1,2] > kp_thre and kps[i2,2] > kp_thre:
            cv2.line(
                kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[i1,2] > kp_thre:
            cv2.circle(
                kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[i2,2] > kp_thre:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_2d_pose(pred, img, kps_line, prefix='vis2dpose', bbox=None):
    if type(img) == str:
        img = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    tmpimg = img.copy().astype(np.uint8)
    tmpkps = np.zeros((3, len(pred)))
    tmpkps[0, :], tmpkps[1, :] = pred[:, 0], pred[:, 1]
    tmpkps[2, :] = 1
    tmpimg = vis_keypoints_with_skeleton(tmpimg, tmpkps, kps_line, bbox)

    now = datetime.now()
    file_name = f'{prefix}_{now.isoformat()[:-7]}_2d_joint.jpg'
    cv2.imwrite(osp.join(cfg.vis_dir, file_name), tmpimg)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def vis_3d_pose(kps_3d, kps_line, file_path='image.png', ax_in=None):
    if kps_3d.shape[1] == 4:
        kps_3d_vis = kps_3d[:,[-1]]
        kps_3d = kps_3d[:, :3]
    else:
        kps_3d_vis = np.ones((len(kps_3d), 1))
        
    if not ax_in:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax_in

    for l in range(len(kps_line)):
        i1 = kps_line[l][0]
        i2 = kps_line[l][1]
        x = np.array([kps_3d[i1, 0], kps_3d[i2, 0]])
        y = np.array([kps_3d[i1, 1], kps_3d[i2, 1]])
        z = np.array([kps_3d[i1, 2], kps_3d[i2, 2]])

        if kps_3d_vis[i1, 0] > 0 and kps_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c='r', linewidth=1)
        if kps_3d_vis[i1, 0] > 0:
            ax.scatter(kps_3d[i1, 0], kps_3d[i1, 2], -kps_3d[i1, 1], c='b', marker='o')
        if kps_3d_vis[i2, 0] > 0:
            ax.scatter(kps_3d[i2, 0], kps_3d[i2, 2], -kps_3d[i2, 1], c='b', marker='o')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('Y axis')
    
    ax.set_xlim3d(-800, 800)
    ax.set_ylim3d(-800, 800)
    ax.set_zlim3d(-800, 800)

    title = '3D Skeleton'
    ax.set_title(title)
    axisEqual3D(ax)

    if not ax_in:
        fig.savefig(file_path)
        plt.close(fig=fig)
    else:
        return ax

def save_obj(v, f=None, file_name=''):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    if f is not None:
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def print_eval_history(eval_history):
    message = f''
    for k in eval_history.keys():
        message += f"{k.upper()}: {eval_history[k][-1]:.2f}\t"
    logger.info(message)


def get_color(idx, num, map='coolwarm'):
    cmap = plt.get_cmap(map)
    colors = [cmap(i) for i in np.linspace(0, 1, num)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    if idx % 2 == 0: return colors[idx//2]
    else: return colors[-(idx//2+1)]


def render_mesh(img, mesh, face, cam_param, color=(0.5, 0.5, 0.4), return_mesh=False):
    mesh = trimesh.Trimesh(mesh, face)
    if 'R' in cam_param:
        mesh.vertices = np.dot(mesh.vertices, cam_param['R'])
    if 't' in cam_param:
        mesh.vertices += cam_param['t'][None,:]

    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    # light
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.25, 0.25, 0.25))
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.85)
    light_pose = np.eye(4)

    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)


    # mesh
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(color[0],color[1],color[2],1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh, 'mesh')

    # camera
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    valid_mask = (depth > 0)[:, :, None]
    img = rgb[:, :, :3] * valid_mask + img * (1 - valid_mask)
    if return_mesh:
        return img.astype(np.uint8), rgb
    else:
        return img.astype(np.uint8)