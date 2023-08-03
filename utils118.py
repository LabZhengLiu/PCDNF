import numpy as np
from sklearn.decomposition import PCA
import math
import torch
import argparse
##########################Parameters########################
#
#
#
#
###############################################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_arguments():
    parser = argparse.ArgumentParser()
    # naming / file handling
    parser.add_argument('--name', type=str, default='pcdenoising', help='training run name')
    parser.add_argument('--network_model_dir', type=str, default='./Models/all/test1', help='output folder (trained models)')
    parser.add_argument('--trainset', type=str, default='./dataset/Train', help='training set file name')
    parser.add_argument('--testset', type=str, default='./Dataset/Test', help='testing set file name')
    parser.add_argument('--save_dir', type=str, default='./Results/all/test1', help='')
    parser.add_argument('--summary_train', type=str, default='.logs/all/test', help='')
    parser.add_argument('--summary_test', type=str, default='./Summary/logs/model1/test', help='')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--manualSeed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--patch_per_shape', type=int, default=8000, help='')
    parser.add_argument('--patch_radius', type=float, default=0.05, help='')
    parser.add_argument('--knn patch',type=bool,default=True,help='use knn neighboorhood to construct patch')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--model_interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')

    # others parameters
    parser.add_argument('--resume', type=str, default='', help='refine model at this path')
    parser.add_argument('--support_multiple', type=float, default=4.0, help='the multiple of support radius')
    parser.add_argument('--support_angle', type=int, default=15, help='')
    parser.add_argument('--gt_normal_mode', type=str, default='nearest', help='')
    parser.add_argument('--repulsion_alpha', type=float, default='0.98', help='')

    # evaluation parameters
    parser.add_argument('--eval_dir', type=str, default='./Models/all/test1', help='')
    parser.add_argument('--eval_iter_nums', type=int, default=3, help='')

    return parser.parse_args()

###################Pre-Processing Tools########################
#
#
#
#
###############################################################


def get_principle_dirs(pts):

    pts_pca = PCA(n_components=3)
    pts_pca.fit(pts)
    principle_dirs = pts_pca.components_
    principle_dirs /= np.linalg.norm(principle_dirs, 2, axis=0)

    return principle_dirs


def pca_alignment(pts, random_flag=False):

    pca_dirs = get_principle_dirs(pts)

    if random_flag:

        pca_dirs *= np.random.choice([-1, 1], 1)

    rotate_1 = compute_roatation_matrix(pca_dirs[2], [0, 0, 1], pca_dirs[1])
    pca_dirs = np.array(rotate_1 * pca_dirs.T).T
    rotate_2 = compute_roatation_matrix(pca_dirs[1], [1, 0, 0], pca_dirs[2])
    pts = np.array(rotate_2 * rotate_1 * np.matrix(pts.T)).T

    inv_rotation = np.array(np.linalg.inv(rotate_2 * rotate_1))

    return pts, inv_rotation

def compute_roatation_matrix(sour_vec, dest_vec, sour_vertical_vec=None):
    # http://immersivemath.com/forum/question/rotation-matrix-from-one-vector-to-another/
    if np.linalg.norm(np.cross(sour_vec, dest_vec), 2) == 0 or np.abs(np.dot(sour_vec, dest_vec)) >= 1.0:
        if np.dot(sour_vec, dest_vec) < 0:
            return rotation_matrix(sour_vertical_vec, np.pi)
        return np.identity(3)
    alpha = np.arccos(np.dot(sour_vec, dest_vec))
    a = np.cross(sour_vec, dest_vec) / np.linalg.norm(np.cross(sour_vec, dest_vec), 2)
    c = np.cos(alpha)
    s = np.sin(alpha)
    R1 = [a[0] * a[0] * (1.0 - c) + c,
          a[0] * a[1] * (1.0 - c) - s * a[2],
          a[0] * a[2] * (1.0 - c) + s * a[1]]

    R2 = [a[0] * a[1] * (1.0 - c) + s * a[2],
          a[1] * a[1] * (1.0 - c) + c,
          a[1] * a[2] * (1.0 - c) - s * a[0]]

    R3 = [a[0] * a[2] * (1.0 - c) - s * a[1],
          a[1] * a[2] * (1.0 - c) + s * a[0],
          a[2] * a[2] * (1.0 - c) + c]

    R = np.matrix([R1, R2, R3])

    return R


def rotation_matrix(axis, theta):

    # Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.

    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.matrix(np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]))




##########################Network Tools########################
#
#
#
#
###############################################################

def adjust_learning_rate(optimizer, epoch, opt):

    lr_shceduler(optimizer, epoch, opt.lr)

def lr_shceduler(optimizer, epoch, init_lr):

    if epoch > 36:
        init_lr *= 0.5e-3
    elif epoch > 32:
        init_lr *= 1e-3
    elif epoch > 24:
        init_lr *= 1e-2
    elif epoch > 16:
        init_lr *= 1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr

################################Ablation Study of Different Loss ###############################

#论文中第一种的方案，La_proj
def compute_original_1_loss(pts_pred, gt_patch_pts, gt_patch_normals, support_radius, alpha):

    pts_pred = pts_pred.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = (pts_pred - gt_patch_pts).pow(2).sum(2)

    # avoid divided by zero
    weight = torch.exp(-1 * dist_square / (support_radius ** 2)) + 1e-12
    weight = weight / weight.sum(1, keepdim=True)

    # key loss
    project_dist = ((pts_pred - gt_patch_pts) * gt_patch_normals).sum(2)
    imls_dist = torch.abs((project_dist * weight).sum(1))

    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]

    # final loss
    dist = torch.mean((alpha * imls_dist) + (1 - alpha) * max_dist)

    return dist
#使用双边滤波
def compute_original_2_loss(pred_point, gt_patch_pts, gt_patch_normals, support_radius, support_angle, alpha):

    # Compute Spatial Weighted Function
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))

    ############# Get The Nearest Normal For Predicted Point #############
    nearest_idx = torch.argmin(dist_square, dim=1)
    pred_point_normal = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(nearest_idx)])
    pred_point_normal = pred_point_normal.view(-1, 3)
    pred_point_normal = pred_point_normal.unsqueeze(1)
    pred_point_normal = pred_point_normal.repeat(1, gt_patch_normals.size(1), 1)
    ############# Get The Nearest Normal For Predicted Point #############

    # Compute Normal Weighted Function
    normal_proj_dist = (pred_point_normal * gt_patch_normals).sum(2)
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - np.cos(support_angle)))**2)

    # # avoid divided by zero
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(1, keepdim=True)

    # key loss
    #不同于poinfilter的地方，Pointfilter用dist_square*normal
    project_dist = torch.sqrt(dist_square)
    imls_dist = (project_dist * weight).sum(1)

    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]

    # final loss
    dist = torch.mean((alpha * imls_dist) + (1 - alpha) * max_dist)

    return dist
#PointCleanNet
def compute_original_3_loss(pts_pred, gt_patch_pts, alpha):
    # PointCleanNet Loss
    pts_pred = pts_pred.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    m = (pts_pred - gt_patch_pts).pow(2).sum(2)
    min_dist = torch.min(m, 1)[0]
    max_dist = torch.max(m, 1)[0]
    dist = torch.mean((alpha * min_dist) + (1 - alpha) * max_dist)
    # print('min_dist: %f max_dist: %f' % (alpha * torch.mean(min_dist).item(), (1 - alpha) * torch.mean(max_dist).item()))
    return dist * 100

def compute_original_4_loss(pts_pred1,pts_pred2, gt_patch_pts,alpha):
    # PointCleanNet Loss
    pts_pred1= pts_pred1.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    m = (pts_pred1 - gt_patch_pts).pow(2).sum(2)
    min_dist = torch.min(m, 1)[0]
    max_dist = torch.max(m, 1)[0]

    dist1 =torch.mean((alpha * min_dist) + (1 - alpha) * max_dist)

    pts_pred2= pts_pred2.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    m = (pts_pred2 - gt_patch_pts).pow(2).sum(2)
    min_dist = torch.min(m, 1)[0]
    max_dist = torch.max(m, 1)[0]
    dist2 = torch.mean((alpha * min_dist) + (1 - alpha) * max_dist)

    dist=dist1+dist2

    # print('min_dist: %f max_dist: %f' % (alpha * torch.mean(min_dist).item(), (1 - alpha) * torch.mean(max_dist).item()))
    return dist * 100

def compute_original_5_loss(pts_pred1,pts_pred2,normal, gt_patch_pts,gt_normal,alpha):
    # PointCleanNet Loss
    Batchsize=gt_patch_pts.size(0)
    pts_pred1= pts_pred1.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    m = (pts_pred1 - gt_patch_pts).pow(2).sum(2)
    min_dist= torch.min(m, 1)[0]
    max_dist = torch.max(m, 1)[0]
    dist1 =torch.mean((alpha * min_dist) + (1 - alpha) * max_dist)


    pred_ponts=pts_pred2
    pts_pred2= pts_pred2.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    m = (pts_pred2 - gt_patch_pts).pow(2).sum(2)
    min_dist,idx= torch.min(m, 1)
    max_dist = torch.max(m, 1)[0]
    dist2 = torch.mean((alpha * min_dist) + (1 - alpha) * max_dist)

    idx=idx.unsqueeze(-1).unsqueeze(-1)
    nearestpoint=torch.gather(gt_patch_pts,dim=1,index=idx.expand(Batchsize,1,3))
    nearestpoint=nearestpoint.squeeze(1)
    point=(pred_ponts-nearestpoint).unsqueeze(-1)
    pointnormal=normal.unsqueeze(1)
    oth=torch.abs(torch.bmm(pointnormal,point))
    oth=oth.mean()*100

    normal_dist=(normal-gt_normal).pow(2).sum(1).mean()
    dist=(dist1+dist2)*100
    out=dist+normal_dist+oth


    # print('min_dist: %f max_dist: %f' % (alpha * torch.mean(min_dist).item(), (1 - alpha) * torch.mean(max_dist).item()))
    return out,dist,normal_dist,oth


def compute_original_6_loss(pts_pred1,gt_patch_pts,normal,gtnormal, alpha):
    # PointCleanNet Loss
    pts_pred1= pts_pred1.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    m = (pts_pred1 - gt_patch_pts).pow(2).sum(2)
    min_dist = torch.min(m, 1)[0]
    max_dist = torch.max(m, 1)[0]

    dist =torch.mean((alpha * min_dist) + (1 - alpha) * max_dist)
    dist=dist*100

    loss1= torch.nn.functional.nll_loss(normal, gtnormal)

    loss=loss1+dist
    # print('min_dist: %f max_dist: %f' % (alpha * torch.mean(min_dist).item(), (1 - alpha) * torch.mean(max_dist).item()))
    return loss,dist,loss1
################################Ablation Study of Different Loss ###############################
#作者改进的双边滤波
def compute_original_7_loss(pts_pred1,gt_patch_pts,normal,gtnormal,patch_center_normal,alpha):
    # PointCleanNet Loss
    Batchsize = gt_patch_pts.size(0)
    pred_ponts = pts_pred1
    pts_pred1= pts_pred1.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    m = (pts_pred1 - gt_patch_pts).pow(2).sum(2)
    min_dist,idx= torch.min(m, 1)
    max_dist = torch.max(m, 1)[0]

    dist =torch.mean((alpha * min_dist) + (1 - alpha) * max_dist)
    dist=dist*100
    '''
    idx = idx.unsqueeze(-1).unsqueeze(-1)
    nearestpoint = torch.gather(gt_patch_pts, dim=1, index=idx.expand(Batchsize, 1, 3))
    nearestpoint = nearestpoint.squeeze(1)
    point = (pred_ponts - nearestpoint).unsqueeze(-1)
    pointnormal = patch_center_normal
    oth = torch.abs(torch.bmm(pointnormal, point))
    oth = oth.mean() * 100
    '''
    #点法向量相乘是否要加系数？
    loss1= torch.nn.functional.nll_loss(normal, gtnormal)

    loss=loss1+dist
    # print('min_dist: %f max_dist: %f' % (alpha * torch.mean(min_dist).item(), (1 - alpha) * torch.mean(max_dist).item()))
    return loss,dist,loss1
def compute_original_8_loss(pred_point, gt_patch_pts, gt_patch_normals,deltnorma,prednormal,support_radius, support_angle, alpha):

    # Our Loss
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))

    nearest_idx = torch.argmin(dist_square, dim=1)
    pred_point_normal = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(nearest_idx)])
    pred_point_normal = pred_point_normal.view(-1, 3)
    pred_point_normal = pred_point_normal.unsqueeze(1)
    pred_point_normal = pred_point_normal.repeat(1, gt_patch_normals.size(1), 1)

    normal_proj_dist = (pred_point_normal * gt_patch_normals).sum(2)
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - np.cos(support_angle)))**2)

    # # avoid divided by zero
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(1, keepdim=True)

    # key loss
    project_dist = torch.abs(((pred_point - gt_patch_pts) * gt_patch_normals).sum(2))
    imls_dist = (project_dist * weight).sum(1)

    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]

    loss1 = torch.nn.functional.nll_loss(prednormal, deltnorma)

    # final loss
    dist = torch.mean((alpha * imls_dist) + (1 - alpha) * max_dist)

    loss=dist+loss1

    return loss
def compute_bilateral_loss(pred_point,pred_normal, gt_patch_pts, gt_patch_normals,predweight,support_radius, support_angle, alpha,top_idx):

    # Our Loss
    # Our Loss
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))

    nearest_idx = torch.argmin(dist_square, dim=1)
    pred_point_normal = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(nearest_idx)])
    pred_point_normal = pred_point_normal.view(-1, 3)
    pred_point_normal = pred_point_normal.unsqueeze(1)
    pred_point_normal = pred_point_normal.repeat(1, gt_patch_normals.size(1), 1)

    normal_proj_dist = (pred_point_normal * gt_patch_normals).sum(2)
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - np.cos(support_angle)))**2)

    # # avoid divided by zero
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(1, keepdim=True)

    # key loss
    project_dist = torch.abs(((pred_point - gt_patch_pts) * gt_patch_normals).sum(2))
    imls_dist = (project_dist * weight).sum(1)

    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]

    # final loss
    loss1 = 100*torch.mean((alpha * imls_dist) + (1 - alpha) * max_dist)

    pred_normal=pred_normal.unsqueeze(1)
    pred_normal=pred_normal.repeat(1,gt_patch_normals.size(1),1)
    loss2=10*(pred_normal-pred_point_normal).pow(2).sum(2).mean(1).mean(0)

    oth_loss = (pred_normal*(pred_point-gt_patch_pts)).sum(2).pow(2)
    oth_loss = 10*(oth_loss).mean()

    loss=loss1+loss2+oth_loss

    return loss,loss1,loss2,oth_loss

def compute_bilateral_loss1(pred_point,pred_normal, gt_patch_pts, gt_patch_normals,predweight,support_radius, support_angle, alpha,top_idx):

    # Our Loss
    # Our Loss
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))

    nearest_idx = torch.argmin(dist_square, dim=1)
    pred_point_normal = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(nearest_idx)])
    pred_point_normal = pred_point_normal.view(-1, 3)
    pred_point_normal = pred_point_normal.unsqueeze(1)
    pred_point_normal = pred_point_normal.repeat(1, gt_patch_normals.size(1), 1)

    normal_proj_dist = (pred_point_normal * gt_patch_normals).sum(2)
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - np.cos(support_angle)))**2)

    # # avoid divided by zero
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(1, keepdim=True)

    # key loss
    project_dist = torch.abs(((pred_point - gt_patch_pts) * gt_patch_normals).sum(2))
    imls_dist = (project_dist * weight).sum(1)

    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]

    # final loss
    loss1 = 100*torch.mean((alpha * imls_dist) + (1 - alpha) * max_dist)

    pred_normal=pred_normal.unsqueeze(1)
    pred_normal=pred_normal.repeat(1,gt_patch_normals.size(1),1)
    loss2=10*(pred_normal-pred_point_normal).pow(2).sum(2).mean(1).mean(0)

    # oth_loss =predweight*(pred_normal*(pred_point-gt_patch_pts)).sum(2).pow(2)
    # oth_loss = 10*(oth_loss).mean()

    loss=loss1+loss2

    return loss,loss1,loss2


def compute_bilateral_loss_with_repulsion(pred_point, gt_patch_pts, gt_patch_normals, support_radius, support_angle, alpha):

    # Our Loss
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))

    nearest_idx = torch.argmin(dist_square, dim=1)
    pred_point_normal = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(nearest_idx)])
    pred_point_normal = pred_point_normal.view(-1, 3)
    pred_point_normal = pred_point_normal.unsqueeze(1)
    pred_point_normal = pred_point_normal.repeat(1, gt_patch_normals.size(1), 1)

    normal_proj_dist = (pred_point_normal * gt_patch_normals).sum(2)
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - np.cos(support_angle)))**2)

    # # avoid divided by zero
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(1, keepdim=True)

    # key loss
    project_dist = torch.abs(((pred_point - gt_patch_pts) * gt_patch_normals).sum(2))
    imls_dist = (project_dist * weight).sum(1)

    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]

    # final loss
    dist = torch.mean((alpha * imls_dist) + (1 - alpha) * max_dist)

    return dist
#loss =compute_original_L2_loss(x, gt_patch,delta_normal,xx,opt.repulsion_alpha)
def compute_original_L2_loss(pts_pred, gt_patch_pts,gt_mask,pred_mask,alpha):
    # PointCleanNet Loss

    #classficaton loss
    loss1=torch.nn.functional.nll_loss(pred_mask,gt_mask)
    #loss1=100*loss1
    loss2=compute_original_3_loss(pts_pred,gt_patch_pts,alpha)
    loss=0.5*loss1+0.5*loss2

    return loss,loss1,loss2

def compute_orginal_Pointfilter_loss(pred_point, gt_patch_pts, gt_patch_normals, support_radius, support_angle,gt_mask,pred_mask,alpha):

    loss1=torch.nn.functional.nll_loss(pred_mask,gt_mask)
    loss2=compute_bilateral_loss_with_repulsion(pred_point,gt_patch_pts,gt_patch_normals,support_radius,support_angle,alpha)
    loss=0.5*loss1+0.5*loss2

    return loss,loss1,loss2
def cos_angle(v1, v2):

    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

def Patch_Normal_loss_Compute(pred_normas,gt_normals,top_idx):

    B,k=top_idx.size()
    gt_normals=torch.gather(gt_normals, dim=1, index=top_idx.unsqueeze(-1).expand(B, k, 3))#[B,256,3]
    # normal_loss=torch.min((pred_normas-gt_normals).pow(2).sum(2),(pred_normas+gt_normals).pow(2).sum(2)).mean(1).mean()
    normal_loss = (pred_normas - gt_normals).pow(2).sum(2).mean(1).mean()

    return normal_loss

def Normal_loss_Compute(pred_normal,gt_normal):
    gt_normal=gt_normal.squeeze(1)
    # normal_loss=torch.min((pred_normas-gt_normals).pow(2).sum(2),(pred_normas+gt_normals).pow(2).sum(2)).mean(1).mean()
    normal_loss = (pred_normal - gt_normal).pow(1).sum(1).mean()

    return normal_loss

def Cos_Compute_Normal_Loss(pre_normals,gt_normals):
    loss=(1 - torch.abs(cos_angle(pre_normals,gt_normals))).pow(2).mean()
    return loss
def Sin_Compute_Normal_Loss(pre_normals,gt_normals):

    loss= 0.5*torch.norm(torch.cross(pre_normals, gt_normals, dim=-1), p=2, dim=1).mean()
    return loss
'''
def Otho_Loss(gt_normals,gt_normal,gt_points,pre_point,index):

    k=index.size(1)
    B=index.size(0)
    gt_normals=torch.gather(gt_normals,dim=1,index=index.unsqueeze(-1).expand(B,k,3))
    gt_points=torch.gather(gt_points,dim=1,index=index.unsqueeze(-1).expand(B,k,3))

    pre_point=pre_point.unsqueeze(-1).repeat(1,1,k).transpose(2,1)
    loss1=(torch.abs(gt_normals*(pre_point-gt_points))).sum(-1).sum(1).mean(0)

    gt_normal=gt_normal.repeat(1,k,1)
    loss2=(torch.abs(gt_normal*(pre_point-gt_points))).sum(-1).sum(1).mean(0)

    loss=loss1+loss2
    return loss
'''
def Otho_Loss(pred_normal,gt_point,pred_point):

    pred_point=pred_point.unsqueeze(1)
    # pred_normal=pred_normal.unsqueeze(1)
    point_constrain=(pred_point-gt_point).pow(2).sum(2).mean(0)

    point_normal=(pred_normal*(pred_point-gt_point)).sum(2).unsqueeze(-1)
    normal_point_constrain=(point_normal*pred_normal).pow(2).sum(2).mean(0)

    constrain=torch.abs(point_constrain-normal_point_constrain)

    return constrain





def compute_loss(pred_point,pred_normal, gt_patch_pts, gt_patch_normals,predweight,gt_normal,support_radius,support_angle,alpha):
    # Our Loss
    orginal_point=pred_point
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)#[B,3]-->[B,N,3]
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)

    nearest_idx = torch.argmin(dist_square, dim=1)
    neareat_point = torch.cat([gt_patch_pts[i, index, :] for i, index in enumerate(nearest_idx)])
    neareat_point = neareat_point.view(-1, 3)#[64,3]

    max_dist = torch.max(dist_square, 1)[0]
    max_dist=torch.mean(max_dist)
    # loss1=10*(torch.abs((orginal_point-neareat_point).pow(2).sum(1)-(pred_normal*(orginal_point-neareat_point)).sum(1).pow(2))).mean()
    gt_normal=gt_normal.squeeze(1)
    # key loss
    pred_normal=pred_normal.unsqueeze(1).repeat(1,gt_patch_pts.size(1),1)
    project_dist =(gt_patch_normals*(pred_point - gt_patch_pts)).sum(2).pow(2)#[b,n]
    normal_dist=(pred_normal*(gt_patch_pts-0)).sum(2).pow(2)
    oth_loss=100*((normal_dist+project_dist)*predweight).sum(1).mean()

    dist=oth_loss+max_dist

    return dist,oth_loss,max_dist

def compute_loss1(pred_point,pred_normal, gt_patch_pts, gt_patch_normals,predweight,gt_normal,support_radius,support_angle,alpha):
    # Our Loss
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)#[B,3]-->[B,N,3]
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)

    min_dist=torch.min(dist_square,1)[0]
    max_dist = torch.max(dist_square, 1)[0]
    # final loss
    loss1 = 100*torch.mean((alpha * min_dist) + (1 - alpha) * max_dist)
    gt_normal=gt_normal.squeeze(1)
    loss2=10*(pred_normal-gt_normal).pow(2).sum(1).mean()
    # key loss
    pred_normal=pred_normal.unsqueeze(1).repeat(1,gt_patch_pts.size(1),1)
    project_dist =(gt_patch_normals*(pred_point - gt_patch_pts)).sum(2).pow(2)#[b,n]
    normal_dist=(pred_normal*(pred_point-gt_patch_pts)).sum(2).pow(2)
    oth_loss=10*((normal_dist+project_dist)*predweight).sum(1).mean()
    # regularizer = - torch.mean(predweight.log())
    dist=0.5*loss1+0.5*loss2+oth_loss


    return dist,loss1,loss2,oth_loss
def compute_ditstance(gt_patch_normals,gt_patch_pts,pred_point):
    dist=(gt_patch_normals*(pred_point-gt_patch_pts)).sum(2)
    return dist
def comtrative_loss(pred_point,pred_normal, gt_patch_pts, gt_patch_normals,topidx,gt_normal,support_radius,support_angle,alpha):
    device = torch.device('cuda')
    B,N,C=gt_patch_pts.size()
    label=torch.zeros(B,N,1,device=device)
    idx_base = torch.arange(0, 64,device=device).view(-1, 1) *N
    topidx=topidx+idx_base
    topidx=topidx.view(-1)
    label=label.view(B*N,-1)
    label[topidx,:]=1
    label=label.view(B,N)
    margin=0.8

    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))

    nearest_idx = torch.argmin(dist_square, dim=1)
    pred_point_normal = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(nearest_idx)])
    pred_point_normal = pred_point_normal.view(-1, 3)
    pred_point_normal = pred_point_normal.unsqueeze(1)
    pred_point_normal = pred_point_normal.repeat(1, gt_patch_normals.size(1), 1)

    normal_proj_dist = (pred_point_normal * gt_patch_normals).sum(2)
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - np.cos(support_angle)))**2)

    # # avoid divided by zero
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(1, keepdim=True)
    # r1=label*(compute_ditstance(gt_patch_normals,gt_patch_pts,pred_point).pow(2))
    # r2=(1-label)*((torch.clamp(margin-compute_ditstance(gt_patch_normals,gt_patch_pts,pred_point),min=0.0)).pow(2))
    loss1=label*(weight*compute_ditstance(gt_patch_normals,gt_patch_pts,pred_point).pow(2))+(1-label)*(weight*(torch.clamp(margin-compute_ditstance(gt_patch_normals,gt_patch_pts,pred_point),min=0.0)).pow(2))
    loss1=1000*loss1.mean()
    gt_normal=gt_normal.squeeze(1)
    loss2 =(pred_normal - gt_normal).pow(2).sum(1).mean()
    loss=10*(loss1+loss2)
    return loss,loss1,loss2


if __name__ == '__main__':

    pred_normal=torch.rand(64,3)
    pred_point=torch.rand(64,3)
    gt_normal=torch.rand(64,3)
    gt_patch_pts=torch.rand(64,512,3)
    gt_patch_normals=torch.rand(64,512,3)
    support_radius=torch.rand(64,1)
    support_angle=0.23898
    alpha=0.97
    predweight=torch.rand(64,512)
    # compute_bilateral_loss_with_repulsion(pred_point,pred_normal,gt_patch_pts,gt_patch_normal,predweight,support_radius,support_angle,alpha)
    compute_loss(pred_point,pred_normal,gt_patch_pts,gt_patch_normals,predweight,gt_normal,alpha)