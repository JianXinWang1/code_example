
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
import os
import torch
from .rays_functions import *
import numpy as np
from .point_clouds_functions import *

class ColmapDataset():
    def __init__(self, root_dir):
        self.downsample = 1.0
        self.root_dir = root_dir
        self.read_intrinsics()
        self.read_meta()

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample

        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self):
        # read extrinsics
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        folder = 'images'

        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)
        self.depth, self.confidence = read_depth_confidence(self.img_wh)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.depth /= scale
        self.pts3d[:,0:3] /= scale
        self.cost_intervial = torch.tensor(np.load('../depth_confidence/mvs_data.npy')).float()
        self.cost_intervial /= scale
        self.bds = [min(self.cost_intervial), max(self.cost_intervial)]
        self.rays = []

        # use every 8th image as test set
        img_paths = [x for x in img_paths]

        depth_confidence_buf = []
        buf = []
        for img_path in img_paths:
            img = read_image(img_path, self.img_wh, blend_a=False)
            img = torch.FloatTensor(img)
            buf += [img]
        self.images = buf


        for i in range(len(img_paths)):
            depth_tem = rearrange(self.train_depth[i], 'h w -> (h w)')
            confidence_tem = rearrange(self.train_confidence[i], 'h w -> (h w)')
            depth_tem = torch.FloatTensor(depth_tem)
            confidence_tem = torch.FloatTensor(confidence_tem)
            dandc = torch.cat((depth_tem[..., None], confidence_tem[..., None]), dim=1)
            depth_confidence_buf.append(dandc)
        self.dandc = torch.stack(depth_confidence_buf)

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)