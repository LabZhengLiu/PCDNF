import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import os
import math


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_idx(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    return idx  # (batch_size, 2*num_dims, num_points, k)


def get_knn_feature(x, k=20):
    idx = get_idx(x, k=k)

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class FeatureExtration(nn.Module):
    def __init__(self, input_dim, output_dim, rate1, rate2, rate3):
        super(FeatureExtration, self).__init__()
        self.bn1_1 = nn.BatchNorm2d(output_dim // rate1)
        self.bn1_2 = nn.BatchNorm2d(output_dim // rate2)
        self.bn1_3 = nn.BatchNorm1d(output_dim // rate3)
        self.bn1_4 = nn.BatchNorm1d(output_dim)
        self.bn1_5 = nn.BatchNorm2d(output_dim // rate3)

        self.conv1_1 = nn.Sequential(nn.Conv2d(input_dim * 2, output_dim // rate1, 1), self.bn1_1,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv1_2 = nn.Sequential(nn.Conv2d(input_dim * 2, output_dim // rate2, 1), self.bn1_2,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv1_3 = nn.Sequential(nn.Conv1d(output_dim // rate1 + output_dim // rate2, output_dim // rate3, 1),
                                     self.bn1_3,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv1_5 = nn.Sequential(nn.Conv2d((output_dim // rate3) * 2, output_dim // rate3, 1), self.bn1_5,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv1_4 = nn.Sequential(nn.Conv1d(output_dim // rate3, output_dim, 1), self.bn1_4,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.fc1 = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // 2, 3)
        )

    def forward(self, point):
        '''

        :param point: [B,3,N]
        :return: feature :[B,N,Outputdim]
                refinepoint:[B,N,3]
        '''
        pointfeature = get_knn_feature(point, k=8)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k))
        pointfeature = self.conv1_1(pointfeature)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        pointfeature1 = pointfeature.max(dim=-1, keepdim=False)[
            0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        pointfeature = get_knn_feature(point, k=16)
        pointfeature = self.conv1_2(pointfeature)
        pointfeature2 = pointfeature.max(dim=-1, keepdim=False)[0]
        pointfeature = torch.cat([pointfeature1, pointfeature2], dim=1)
        pointfeature = self.conv1_3(pointfeature)

        pointfeature = get_knn_feature(pointfeature, k=16)
        pointfeature = self.conv1_5(pointfeature)
        pointfeature = pointfeature.max(dim=-1, keepdim=False)[0]

        pointfeature = self.conv1_4(pointfeature)

        pointfeature = pointfeature.transpose(2, 1)
        refinepoint = self.fc1(pointfeature)
        refinepoint = refinepoint + point.transpose(2, 1)

        return pointfeature, refinepoint


class ConsistentPointSelect(nn.Module):
    def __init__(self, r=0.5):
        super(ConsistentPointSelect, self).__init__()
        self.r = r

        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
        )
        self.bn1 = nn.BatchNorm1d(128)
        self.conv1 = nn.Sequential(nn.Conv1d(192, 128, 1), self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.sig=nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

    def angle(self, v1, v2):
        cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                                  v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                                  v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
        cross_prod_norm = torch.norm(cross_prod, dim=-1)
        dot_prod = torch.sum(v1 * v2, dim=-1)
        result = torch.atan2(cross_prod_norm, dot_prod)
        result = result.unsqueeze(-1)
        return result

    def get_center_normal(self, normalfeature, idx):
        # idx=np.load('top.npy')
        # print(idx.shape)
        # idx = torch.from_numpy(idx)
        B, N, C = normalfeature.size()
        center_normal = torch.gather(normalfeature, dim=1, index=idx.unsqueeze(-1).expand(B, 1, C))
        center_normal = center_normal.repeat(1, N, 1)
        normalfeature = normalfeature - center_normal
        # normalfeature=F.normalize(normalfeature,dim=2)
        # normalfeature=torch.exp(-torch.abs(normalfeature))
        return normalfeature

    def forward(self, pointfea, normalfea, index, point, normal):
        '''

        :param pointfea: point-wise feature [B,N,C]
        :param normal: normal-wise feature [B,N,C]
        :param index: refine center normal position[B,1]
        :param point: point coordinate [B,N,3]
        :param normal: normal coordinate [B,N,3]
        :return:
                 topidx [B,k]
                 keypointfeature[B,k,C]
                 keypoint[B,k,3]
                 keynormalfeature[B,k,C]
                 keynormal[B,k,3]
        '''
        B, N, C = pointfea.size()
        k = int(self.r * N)
        # ||xi-xj||,???不确定保留
        distance = point * point
        pointdist = torch.sum(distance, dim=-1, keepdim=True)

        pointdist = torch.exp(-pointdist)
        pointdist = self.fc4(pointdist)

        angle = self.angle(point, normal)
        angle = self.fc3(angle)

        pointfeature = self.fc1(pointfea)
        # normalfeature=self.get_center_normal(normalfea,index)
        normalfeature = normalfea
        normalfeature = self.fc2(normalfeature)

        feature = torch.cat([pointfeature, normalfeature, angle, pointdist], dim=2)
        feature = feature.transpose(2, 1)
        feature = self.conv1(feature)
        feature = feature.transpose(2, 1)  # [B,N,C]
        feature = torch.max(feature, dim=-1)[0]  # [B,N]
        weight = self.sig(feature)  # [B,N]
        top_idx = torch.argsort(weight, dim=-1, descending=True)[:, 0:k]

        keypointfeature = torch.gather(pointfea, dim=1, index=top_idx.unsqueeze(-1).expand(B, k, C))
        keynormalfeature = torch.gather(normalfea, dim=1, index=top_idx.unsqueeze(-1).expand(B, k, C))
        keyrefinepoint = torch.gather(point, dim=1, index=top_idx.unsqueeze(-1).expand(B, k, 3))
        keyrefinenormal = torch.gather(normal, dim=1, index=top_idx.unsqueeze(-1).expand(B, k, 3))

        return weight, top_idx, keypointfeature, keyrefinepoint, keynormalfeature, keyrefinenormal


class KeyFeatureFusion(nn.Module):
    def __init__(self):
        super(KeyFeatureFusion, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.t = nn.Conv1d(128, 64, 1)
        # linear transform to get keys
        self.p = nn.Conv1d(128, 64, 1)
        # linear transform to get query
        self.g = nn.Conv1d(128, 128, 1)
        self.z = nn.Conv1d(256, 256, 1)

        self.gn = nn.GroupNorm(num_groups=1, num_channels=256)

        self.softmax = nn.Softmax(dim=-1)

    def normalAttention(self, points, normals):
        # print(points.shape)
        # print(normals.shape)
        t = self.t(points)  # [batchsize,64,500]
        p = self.p(points)  # [batchsize,64,500]
        v = self.g(normals)
        proj_query = t  # B X C/2 XN

        proj_key = p.transpose(2, 1)  # B X M XC/2

        energy = torch.bmm(proj_key, proj_query)  # [B,N,N]

        total_energy = energy
        attention = self.softmax(total_energy)  # B X N X N
        # print(attention.shape)
        proj_value = v
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # print(out.shape)
        return out

    def knnfeature(self, x, normalfvals, k):
        '''

        :param x: x is normal/point cardinate [B,N,3]
        :param normalfvals: normalfvals is normal/point feature [B,C,N]
        :param k: K neighbors
        :return: k normal features [B,N,K,C]
        '''
        x = x.transpose(2, 1).contiguous()
        batch_size, num_points, num_dims = normalfvals.size()
        idx = get_idx(x, k=k)
        normalfvals = normalfvals.transpose(2,
                                            1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = normalfvals.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)

        return feature

    def featurefuse(self, knnpointfeature, keyfeature, topidx):
        B, N, K, C = knnpointfeature.size()
        k = topidx.size(1)
        keyknnfeature = torch.gather(knnpointfeature, dim=1,
                                     index=topidx.unsqueeze(-1).unsqueeze(-1).expand(B, k, K, C))
        # keyfeature=keyfeature.unsqueeze(-1)
        # keyfeature=keyfeature.view(B,k,1,C).repeat(1,1,K,1)
        # keypoint=keypoint.unsqueeze(-1)
        # keypoint=keypoint.view(B,k,1,3).repeat(1,1,K,1)
        # feature is included:[point coordinate,key point feature,key point's knn feature]
        # feature=torch.cat([keyfeature,keyknnfeature],dim=-1)#[B,k,K,C]
        # feature=torch.mean(feature,dim=2)
        keyknnfeature = torch.mean(keyknnfeature, dim=2)  # [B,k,C]
        # keyknnfeature = torch.sum(keyknnfeature, dim=2)  # [B,k,C]
        feature = keyfeature + keyknnfeature
        return feature

    def forward(self, weight, allfeature, keyfeature, refinepoint, keypoint, topidx, k):
        '''

        :param allfeature: [B,N,C]
        :param keyfeature: [B,k,C]
        :param refinepoint: [B,N,3]
        :param keypoint: [B,k,3]
        :param topidx: [B,k,1]
        :param k: knn neighboorhood
        :return: keyknnfeature [B,C,N]
        '''

        # pointfeature=pointfeature.transpose(2,1)
        allfeature = allfeature * weight.unsqueeze(-1)
        knnpointfeature = self.knnfeature(refinepoint, allfeature, k)  # [B,N,K,C]
        feature = self.featurefuse(knnpointfeature, keyfeature, topidx)
        feature = feature.transpose(2, 1)
        feature = self.conv(feature)

        return feature


class NormalEncorder(nn.Module):
    def __init__(self):
        super(NormalEncorder, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2))
        # self.conv1_2=nn.Sequential(nn.Conv1d(128,256,1),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2))
        self.conv1_2 = nn.Sequential(nn.Conv2d(128 * 2, 64, 1), nn.BatchNorm2d(64),
                                     nn.LeakyReLU(negative_slope=0.2))

        # self.conv2_1=nn.Sequential(nn.Conv1d(256,128,1),nn.BatchNorm1d(128),nn.LeakyReLU(negative_slope=0.2))
        # self.conv2_2=nn.Sequential(nn.Conv1d(128,256,1),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2))
        self.conv2_1 = nn.Sequential(nn.Conv2d(64 * 2, 128, 1), nn.BatchNorm2d(128),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv2_2 = nn.Sequential(nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2))

        self.fc1 = nn.Sequential(nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.2))
        self.fc2 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.fc3 = nn.Sequential(nn.Conv2d(64, 3, 1))
        # self.fc3=nn.Linear(64,3)

        self.fc1_1 = nn.Sequential(nn.Linear(256, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2))
        self.fc2_1 = nn.Sequential(nn.Linear(128, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2))
        self.fc3_1 = nn.Sequential(nn.Linear(64, 3, 1))

    def forward(self, x, normalfeature,pointfusefeature):
        # [B,256,N]
        # x=torch.cat([x,normalfeature],dim=1)
        x = x + normalfeature
        x=torch.cat([x,pointfusefeature],dim=1)

        feature = self.conv1_1(x)
        # feature1=self.conv1_2(feature1)
        # feature1=feature1+x

        feature1 = get_knn_feature(feature, k=8)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k))
        feature1 = self.conv1_2(feature1)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        feature1 = feature1.max(dim=-1, keepdim=False)[
            0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        feature1 = get_knn_feature(feature1, k=8)
        feature1 = self.conv2_1(feature1)
        feature1 = feature1.max(dim=-1, keepdim=False)[0]  # [B,128,N]

        feature = feature + feature1
        feature = self.conv2_2(feature)

        dis = feature.max(dim=-1, keepdim=False)[0]
        dis = self.fc1_1(dis)
        dis = self.fc2_1(dis)
        dis = self.fc3_1(dis)
        # feature=F.normalize(feature,p=2)

        return dis


'''
class NormalEncorder(nn.Module):
    def __init__(self):
        super(NormalEncorder,self).__init__()

        self.conv1_1=nn.Sequential(nn.Conv1d(256,1024, 1),nn.BatchNorm1d(1024),nn.LeakyReLU(negative_slope=0.2))
        # self.conv1_2=nn.Sequential(nn.Conv1d(128,256,1),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2))
        # self.mappool1=nn.MaxPool1d(1,stride=2)

        # self.conv2_1=nn.Sequential(nn.Conv1d(256,128,1),nn.BatchNorm1d(128),nn.LeakyReLU(negative_slope=0.2))
        # self.conv2_2=nn.Sequential(nn.Conv1d(128,256,1),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2))


        # self.fc1=nn.Sequential(nn.Linear(256,128),nn.BatchNorm1d(128),nn.LeakyReLU(negative_slope=0.2))
        # self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2))
        # self.fc3=nn.Linear(64,3)

        self.fc1=nn.Sequential(nn.Linear(1024,512),nn.BatchNorm1d(512),nn.LeakyReLU(negative_slope=0.2))
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2))
        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2))
        self.fc=nn.Linear(64,3)



    def forward(self,x,globalfeature,normalfeature):

        x=x+globalfeature
        x=torch.cat([x,normalfeature],dim=1)


        feature1=self.conv1_1(x)
        feature1=self.conv1_2(feature1)
        feature1=feature1+x
        feature1=self.mappool1(feature1)

        feature2=self.conv2_1(feature1)
        feature2=self.conv2_2(feature2)
        feature2=feature2+feature1

        feature=feature2.max(dim=-1,keepdim=False)[0]
        feature=self.fc1(feature)
        feature=self.fc2(feature)
        feature=self.fc3(feature)


        feature=self.conv1_1(x)
        feature=feature.max(dim=-1,keepdim=False)[0]
        feature=self.fc1(feature)
        feature=self.fc2(feature)
        feature=self.fc3(feature)
        feature=torch.tanh(self.fc(feature))

         return feature
'''


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.conv = nn.Sequential(nn.Conv1d(384, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.2))
        self.fc1_1 = nn.Linear(512, 256)
        self.fc1_2 = nn.Linear(256, 64)
        self.fc1_3 = nn.Linear(64, 3)
        self.bn1_11 = nn.BatchNorm1d(256)
        self.bn1_22 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = F.relu(self.bn1_11(self.fc1_1(x)))
        x = F.relu(self.bn1_22(self.fc1_2(x)))
        x = torch.tanh(self.fc1_3(x))
        return x


class PCPNet(nn.Module):
    def __init__(self, num_points=500, output_dim=3, k=20):
        super(PCPNet, self).__init__()
        self.num_points = num_points
        self.k = k

        self.pointfeatEX = FeatureExtration(input_dim=3, output_dim=128, rate1=8, rate2=4, rate3=2)
        self.normalfeatEX = FeatureExtration(input_dim=3, output_dim=128, rate1=8, rate2=4, rate3=2)
        self.weight = ConsistentPointSelect(r=0.5)
        self.pointFeaFu = KeyFeatureFusion()
        self.normalFeaFu = KeyFeatureFusion()
        self.normalDecoder = NormalEncorder()

        self.mlp1 = MLP()

    def forward(self, x, normal, index):
        '''

        :param x: point coordinate [64,3,N]
        :param normal: normal coordinate [64,3,n]
        :param normal_center: patch center coordinate [64,1]
        :return: point,normal
        '''
        # print("here")
        pointfeature, refinepoint = self.pointfeatEX(x)
        normalfeature, refinenormal = self.normalfeatEX(normal)
        weight, topidx, keypointfeature, keypoint, keynormalfeature, keynormal = self.weight(pointfeature,
                                                                                             normalfeature, index,
                                                                                             refinepoint, refinenormal)

        pointfusefeature = self.pointFeaFu(weight, pointfeature, keypointfeature, refinepoint, keypoint, topidx,
                                           k=10)  # [B,C,N]
        # normalfusefeature=self.normalFeaFu(weight,normalfeature,keynormalfeature,refinenormal,keynormal,topidx,k=10)#[B,C,N]
        normalfusefeature = self.normalFeaFu(weight, normalfeature, keynormalfeature, refinepoint, keypoint, topidx,
                                             k=10)

        N = pointfusefeature.size(2)

        globalnormalfeature = torch.max(normalfeature, dim=1, keepdim=True)[0]
        globalnormalfeature = globalnormalfeature.repeat(1, N, 1)
        globalnormalfeature = globalnormalfeature.transpose(2, 1)

        globalpointfeature = torch.max(pointfeature, dim=1, keepdim=True)[0]
        globalpointfeature = globalpointfeature.repeat(1, N, 1)
        globalpointfeature = globalpointfeature.transpose(2, 1)

        maxpointfeature = torch.max(pointfusefeature, dim=2, keepdim=True)[0]
        maxpointfeature = maxpointfeature.repeat(1, 1, N)

        maxnormalfeature = torch.max(normalfusefeature, dim=2, keepdim=True)[0]
        maxnormalfeature = maxnormalfeature.repeat(1, 1, N)  # [B,128,N]

        pfeat = torch.cat([pointfusefeature, globalpointfeature, normalfusefeature], dim=1)
        # nfeat=torch.cat([normalfusefeature,globalnormalfeature],dim=1)

        p = self.mlp1(pfeat)
        normal = self.normalDecoder(normalfusefeature, globalnormalfeature,pointfusefeature)
        # n=self.mlp2(nfeat,False)
        n = normal

        return p, n, weight, topidx


if __name__ == '__main__':
    batchsize = 64
    point = torch.rand(64, 512, 3)
    point = point.transpose(2, 1)
    normal = torch.rand(64, 512, 3)
    normal = normal.transpose(2, 1)
    pfeat = torch.rand(64, 128, 512)
    nfeat = torch.rand(64, 128, 512)
    pdist = torch.rand(64, 1, 512)
    nrag = torch.rand(64, 1, 512)
    index = np.random.randint(10, 20, 64)
    # index=np.expand_dims(index,axis=1)
    # print(index)
    net = PCPNet()
    net(point, normal, index)