from kornia import create_meshgrid
import torch
import os
import cv2
import numpy as np
import re

def get_ray_directions(H, W, K):
    grid = create_meshgrid(H, W, False)[0]
    u, v = grid.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    return directions

def read_depth_confidence(img_wh):
    path = r'colmap_data/scan1/images'
    count = 0
    for file in os.listdir(path):
        count = count + 1
    depth = []
    confidence = []
    for i in range(count):
        if i > 9:
            s = '000000'
        else: s = '0000000'
        d = cv2.resize(read_disp(r'/../depth_confidence/depth_est/' + s + str(i) + '.pfm'), img_wh)
        c = cv2.resize(read_disp(r'../depth_confidence/confidence/' + s + str(i) + '.pfm'), img_wh)
        c[c>1] == 1
        depth.append(d[None,...])
        confidence.append(c[None, ...])

    depth = np.concatenate(depth)
    confidence = np.concatenate(confidence)
    return depth, confidence


def read_disp(filename):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
    else:
        raise Exception('Invalid disparity file format!')
    return disp  # [H, W]

def _read_pfm(file):
    file = open(file, 'rb')
    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def center_poses(poses, pts3d=None):
    pose_avg = average_poses(poses, pts3d) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    if pts3d is not None:
        pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T
        return poses_centered, pts3d_centered

    return poses_centered


def normalize(v):

    return v/np.linalg.norm(v)


def average_poses(poses, pts3d=None):
    # 1. Compute the center
    center = poses[..., 3].mean(0)
    z = normalize(poses[..., 2].mean(0))

    y_ = poses[..., 1].mean(0)

    x = normalize(np.cross(y_, z))

    y = np.cross(z, x)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def read_image(img_path, img_wh, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32)/255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4: # blend A to RGB
        if blend_a:
            img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
        else:
            img = img[..., :3]*img[..., -1:]

    img = cv2.resize(img, img_wh)
    # img = img[0:-1:4,0:-1:4,:]
    img = rearrange(img, 'h w c -> (h w) c')
    return img

