from __future__ import print_function

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

import os
import numpy as np
import scipy.spatial as sp

from utils118 import pca_alignment


##################################New Dataloader Class###########################

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2 ** 32 - 1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape,
                                                                  self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(
            self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class PointcloudPatchDataset(data.Dataset):

    def __init__(self, root=None, shapes_list_file=None, patch_radius=0.05, points_per_patch=512,
                 seed=None, train_state='train', shape_name=None, identical_epoches=False,knn=False):

        self.root = root
        self.shapes_list_file = shapes_list_file

        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.seed = seed
        self.train_state = train_state
        self.identical_epochs = identical_epoches
        self.knn=knn

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2 ** 10 - 1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.shape_patch_count = []
        self.patch_radius_absolute = []
        self.gt_shapes = []
        self.noise_shapes = []

        self.shape_names = []
        if self.train_state == 'evaluation' and shape_name is not None:
            pts_normal = np.load(os.path.join(self.root, shape_name + '.npy'))
            noise_pts = pts_normal[:, 0:3]
            noise_normal = pts_normal[:, 3:6]
            noise_kdtree = sp.cKDTree(noise_pts)
            self.noise_shapes.append(
                {'noise_pts': noise_pts, 'noise_kdtree': noise_kdtree, 'noise_normal': noise_normal})
            self.shape_patch_count.append(noise_pts.shape[0])
            bbdiag = float(np.linalg.norm(noise_pts.max(0) - noise_pts.min(0), 2))
            self.patch_radius_absolute.append(bbdiag * self.patch_radius)
        elif self.train_state == 'train':
            with open(os.path.join(self.root, self.shapes_list_file)) as f:
                self.shape_names = f.readlines()
            self.shape_names = [x.strip() for x in self.shape_names]
            self.shape_names = list(filter(None, self.shape_names))
            for shape_ind, shape_name in enumerate(self.shape_names):
                print('getting information for shape %s' % shape_name)
                if shape_ind % 6 == 0:
                    gt_pts_normal = np.load(os.path.join(self.root, shape_name + '.npy'))
                    gt_pts = gt_pts_normal[:, 0:3]
                    gt_normal = gt_pts_normal[:, 3:6]
                    gt_kdtree = sp.cKDTree(gt_pts)
                    self.gt_shapes.append({'gt_pts': gt_pts, 'gt_normal': gt_normal, 'gt_kdtree': gt_kdtree})
                    self.noise_shapes.append(
                        {'noise_pts': gt_pts, 'noise_kdtree': gt_kdtree, 'noise_normal': gt_normal})
                    noise_pts = gt_pts
                else:

                    pts_normal = np.load(os.path.join(self.root, shape_name + '.npy'))
                    noise_pts = pts_normal[:, 0:3]
                    noise_normal = pts_normal[:, 3:6]
                    noise_kdtree = sp.cKDTree(noise_pts)
                    self.noise_shapes.append(
                        {'noise_pts': noise_pts, 'noise_kdtree': noise_kdtree, 'noise_normal': noise_normal})

                self.shape_patch_count.append(noise_pts.shape[0])
                bbdiag = float(np.linalg.norm(noise_pts.max(0) - noise_pts.min(0), 2))
                self.patch_radius_absolute.append(bbdiag * self.patch_radius)

    def patch_sampling(self, patch_inds):

        if self.identical_epochs:
            self.rng.seed(self.seed)

        # if patch_pts.shape[0] > self.points_per_patch:
        #
        #     sample_index = self.rng.choice(range(patch_pts.shape[0]), self.points_per_patch, replace=False)
        #
        # else:
        #
        #     sample_index = self.rng.choice(range(patch_pts.shape[0]), self.points_per_patch)
        # point_count = min(self.points_per_patch, len(patch_inds))
        if len(patch_inds)>=self.points_per_patch:
            patch_inds = patch_inds[self.rng.choice(len(patch_inds), self.points_per_patch, replace=False)]
        else:
            patch_inds=patch_inds[self.rng.choice(len(patch_inds),self.points_per_patch)]

        return patch_inds

    def gauss_fcn(self,x, mu=0, sigma2=0.12):
        tmp = -(x - mu) ** 2 / (2 * sigma2)

        return np.exp(tmp)


    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)
        noise_shape = self.noise_shapes[shape_ind]
        patch_radius = self.patch_radius_absolute[shape_ind]
        # For noise_patch

        if self.knn:
            #索引中包含中心点
            dist,noise_patch_idx=np.array(noise_shape['noise_kdtree'].query(noise_shape['noise_pts'][patch_ind],self.points_per_patch))
            # patch_radius=dist[-1]
            noise_patch_idx=noise_patch_idx.astype(np.int)
            # print(noise_patch_idx)
        else:
            #索引中不包含中心点
            noise_patch_idx = noise_shape['noise_kdtree'].query_ball_point(noise_shape['noise_pts'][patch_ind],patch_radius)
            #noise_patch_idx=noise_patch_idx.astype(np.int)
            noise_patch_idx=np.array(noise_patch_idx)

        if len(noise_patch_idx) < 3:
            return None

        noise_sample_idx = self.patch_sampling(noise_patch_idx)
        index=np.where(noise_sample_idx==patch_ind)
        index=index[0]

        noise_patch_pts = noise_shape['noise_pts'][noise_sample_idx] - noise_shape['noise_pts'][patch_ind]
        # 返回旋转后的patch，以及逆矩阵R^-1
        noise_patch_pts /= patch_radius
        noise_patch_pts, noise_patch_inv = pca_alignment(noise_patch_pts)

        support_radius = np.linalg.norm(noise_patch_pts.max(0) - noise_patch_pts.min(0), 2) / noise_patch_pts.shape[0]
        support_radius = np.expand_dims(support_radius, axis=0)

        normal=noise_shape['noise_normal'][patch_ind]
        normal=np.expand_dims(normal,axis=0)
        normal = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(normal.T)).T


        noise_patch_normal = noise_shape['noise_normal'][noise_sample_idx]
        noise_patch_normal = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(noise_patch_normal.T)).T

        if self.train_state == 'evaluation':
            return torch.from_numpy(noise_patch_pts), torch.from_numpy(noise_patch_inv), \
                   noise_shape['noise_pts'][patch_ind],torch.from_numpy(noise_patch_normal),torch.from_numpy(index),normal

        # For gt_patch
        gt_shape = self.gt_shapes[shape_ind // 6]
        if self.knn:
        # gt_patch_idx = gt_shape['gt_kdtree'].query_ball_point(noise_shape['noise_pts'][patch_ind], patch_radius)
            dist,gt_patch_idx=gt_shape['gt_kdtree'].query(noise_shape['noise_pts'][patch_ind],self.points_per_patch)
            gt_patch_idx=gt_patch_idx.astype(np.int)
        else:
            gt_patch_idx=np.array(gt_shape['gt_kdtree'].query_ball_point(noise_shape['noise_pts'][patch_ind],patch_radius))
        # print(gt_patch_idx)
        if len(gt_patch_idx) < 3:
            return None

        gt_sample_idx=self.patch_sampling(gt_patch_idx)
        # Patch归一化处理
        gt_patch_pts=gt_shape['gt_pts'][gt_sample_idx]-noise_shape['noise_pts'][patch_ind]
        gt_patch_pts /= patch_radius
        gt_patch_pts = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(gt_patch_pts.T)).T
        # 对patch随机选取500个点
        gt_normal=gt_shape['gt_normal'][patch_ind]
        gt_normal=np.expand_dims(gt_normal,axis=0)
        gt_normal = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(gt_normal.T)).T

        gt_patch_normal=gt_shape['gt_normal'][gt_sample_idx]
        gt_patch_normal = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(gt_patch_normal.T)).T

        gt_point=gt_shape['gt_pts'][patch_ind]
        gt_point=gt_point-noise_shape['noise_pts'][patch_ind]
        gt_point=np.expand_dims(gt_point,axis=0)
        gt_point=np.array(np.linalg.inv(noise_patch_inv)*np.matrix(gt_point.T)).T

        return torch.from_numpy(noise_patch_pts), torch.from_numpy(gt_patch_pts), torch.from_numpy(noise_patch_normal),torch.from_numpy(gt_patch_normal),torch.from_numpy(support_radius),torch.from_numpy(gt_normal),torch.from_numpy(index),torch.from_numpy(normal)

    def __len__(self):
        return sum(self.shape_patch_count)

    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if (index >= shape_patch_offset) and (index < shape_patch_offset + shape_patch_count):
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind


if __name__ == '__main__':
    seed = 3627473
    train_dataset = PointcloudPatchDataset(
        root='./dataset',
        shapes_list_file='train.txt',
        seed=seed,
        train_state='train',
        identical_epoches=True,
        knn=True)
    train_dataset.__getitem__(index=100000)
    # train_datasampler = RandomPointcloudPatchSampler(
    #     train_dataset,
    #     patches_per_shape=8000,
    #     seed=3627473,
    #     identical_epochs=False)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     collate_fn=my_collate,
    #     sampler=train_datasampler,
    #     shuffle=(train_datasampler is None),
    #     batch_size=64,
    #     num_workers=4,
    #     pin_memory=True)
    # for batch_ind, data_tuple in enumerate(train_dataloader):
    #
    #     noise_patch, gt_patch, patch_normal, gt_patch_normal = data_tuple
