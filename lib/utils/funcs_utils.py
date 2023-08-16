import numpy as np
import torch
import math
import cv2
import trimesh
from torch.nn import functional as F
import torchgeometry as tgm
from skimage.util.shape import view_as_windows

from core.config import cfg

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB': img = img[:,:,::-1]
    img = img.astype(np.float32)
    return img
    

def add_pelvis_and_neck(joint_coord, joint_valid, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    
    if joint_valid[lhip_idx] > 0 and joint_valid[rhip_idx] > 0:
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))
        joint_valid = np.append(joint_valid, 1)
    else:
        pelvis = np.zeros_like(joint_coord[0, None, :])
        joint_valid = np.append(joint_valid, 0)

    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')

    if joint_valid[lshoulder_idx] > 0 and joint_valid[rshoulder_idx] > 0:
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1,-1))
        joint_valid = np.append(joint_valid, 1)
    else:
        neck = np.zeros_like(joint_coord[0, None, :])
        joint_valid = np.append(joint_valid, 0)

    joint_coord = np.concatenate((joint_coord, pelvis, neck))
    return joint_coord, joint_valid

def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped

def box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale

def center_scale_to_box(center, scale, pixel_std=200.0):
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
    return bbox

def get_bbox(joint_img, joint_valid, extend_ratio=1.2):

    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, input_shape, target_shape, expand_ratio=1.25, do_sanitize=False):
    if do_sanitize:
        # sanitize bboxes
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((input_shape[1] - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((input_shape[0] - 1, y1 + np.max((0, h - 1))))
        if w*h > 0 and x2 > x1 and y2 > y1:
            bbox = np.array([x1, y1, x2-x1, y2-y1])
        else:
            return None
    
    # aspect ratio preserving bbox
    bbox = np.array(bbox)
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = target_shape[1] / target_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*expand_ratio
    bbox[3] = h*expand_ratio
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def pixel2cam(coords, c, f):
    cam_coord = np.zeros((len(coords), 3))
    z = coords[..., 2].reshape(-1, 1)

    cam_coord[..., :2] = (coords[..., :2] - c) * z / f
    cam_coord[..., 2] = coords[..., 2]
    return cam_coord

def generate_joint_heatmap(joints, joints_vis, image_size, heatmap_size, sigma=2):
    num_joints = joints.shape[0]
    image_size, heatmap_size = np.array(image_size[::-1]), np.array(heatmap_size[::-1])
    target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    target_weight = joints_vis[:,None]
    
    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight.reshape(-1)

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def heatmap_to_coords(batch_heatmaps):
    coords, maxvals = get_max_preds(batch_heatmaps)
    maxvals = maxvals.reshape(maxvals.shape[0], maxvals.shape[1])
    
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array(
                    [
                        hm[py][px+1] - hm[py][px-1],
                        hm[py+1][px]-hm[py-1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25
    
    coords = coords.copy()
    coords_valid = (maxvals > 0.5) * 1.0
    return coords, coords_valid

def image_bound_check(coord, image_size, val=None):
    if val is None:
        val = np.ones((len(coord),))
    else:
        val = val.copy()
    
    idxs = np.logical_or(coord[:,0] < 0, coord[:,0] > image_size[1])
    val[idxs] = 0
    
    idxs = np.logical_or(coord[:,1] < 0, coord[:,1] > image_size[0])
    val[idxs] = 0
    return val


def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1) # 3x3 rotation matrix
    
    rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).to(rot_mat.device).float()],2) # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def axis_angle_to_6d(x):
    x = batch_rodrigues(x.reshape(-1, 3))
    x = rotmat_to_6d(x).reshape(-1, 6)
    return x

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def rotmat_to_6d(poses):
    curr_pose = poses.reshape(-1, 3, 3)
    orth6d = curr_pose[:, :, :2].reshape(-1, 6)
    orth6d = orth6d.reshape(poses.shape[0], -1, 6)
    return orth6d


def split_into_chunks(vid_names, seqlen, stride):
    video_start_end_indices = []
    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
    
    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        video_start_end_indices += start_finish

    return video_start_end_indices


def select_vid(db, target_vid=''):
    valid_names = db['vid_name']
    unique_names = np.unique(valid_names)
    for u_n in unique_names:
        if not target_vid in u_n:
            continue

        indexes = valid_names == u_n

        new_db = {}
        for k,v in db.items():
            new_db[k] = db[k][indexes]
       
    return new_db 


def get_sequence(start_index, end_index, seqlen, data):
    if start_index != end_index:
        return data[start_index:end_index+1].copy()
    else:
        return data[start_index:start_index+1].repeat(seqlen, axis=0).copy()

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def convert_focal_princpt(bbox=None):
    focal = [cfg.CAMERA.focal[0] / cfg.HMR.input_img_shape [1] * bbox[2], cfg.CAMERA.focal[1] / cfg.HMR.input_img_shape[0] * bbox[3]]
    princpt = [cfg.CAMERA.princpt[0] / cfg.HMR.input_img_shape [1] * bbox[2] + bbox[0], cfg.CAMERA.princpt[1] / cfg.HMR.input_img_shape [0] * bbox[3] + bbox[1]]
    return focal, princpt

def slide_window_to_sequence(slide_window,window_step=1):
    window_size = slide_window.shape[1]
    output_len=(slide_window.shape[0]-1)*window_step+window_size
    sequence = [[] for i in range(output_len)]

    for i in range(slide_window.shape[0]):
        for j in range(window_size):
            sequence[i * window_step + j].append(slide_window[i, j, ...])

    for i in range(output_len):
        sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0)

    sequence = torch.stack(sequence)
    return sequence