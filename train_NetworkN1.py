# coding=utf-8

from __future__ import print_function
from tensorboardX import SummaryWriter
from NetworkNN import PCPNet
from dataset923 import PointcloudPatchDataset, RandomPointcloudPatchSampler, my_collate
from utils118  import parse_arguments, adjust_learning_rate,compute_bilateral_loss

import os
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train(opt):
    print(opt)
    if not os.path.exists(opt.summary_train):
        os.makedirs(opt.summary_train)
    if not os.path.exists(opt.network_model_dir):
        os.makedirs(opt.network_model_dir)
    print("Random Seed: ", opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    train_dataset = PointcloudPatchDataset(
        root=opt.trainset,
        shapes_list_file='train.txt',
        patch_radius=0.05,
        seed=opt.manualSeed,
        identical_epoches=False,
        knn=True)
    train_datasampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=8000,
        seed=opt.manualSeed,
        identical_epochs=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=my_collate,
        sampler=train_datasampler,
        shuffle=(train_datasampler is None),
        batch_size=opt.batchSize,
        num_workers=int(opt.workers),
        pin_memory=True)
    num_batch = len(train_dataloader)
    print(num_batch)
    # optionally resume from a checkpoint
    denoisenet =PCPNet()
    denoisenet.cuda()
    optimizer = optim.SGD(
        denoisenet.parameters(),
        lr=opt.lr,
        momentum=opt.momentum)
    train_writer = SummaryWriter(opt.summary_train)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch']
            denoisenet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    for epoch in range(opt.start_epoch, opt.nepoch):
        adjust_learning_rate(optimizer, epoch, opt)
        print('lr is %.10f' % (optimizer.param_groups[0]['lr']))
        for batch_ind, data_tuple in enumerate(train_dataloader):
            denoisenet.train()
            optimizer.zero_grad()
            noise_patch, gt_patch,patch_normal,gt_patch_normal,support_radius,gt_normal,index,normal= data_tuple
            noise_patch = noise_patch.float().cuda()
            gt_patch = gt_patch.float().cuda()
            patch_normal=patch_normal.float().cuda()
            gt_patch_normal=gt_patch_normal.float().cuda()
            support_radius = opt.support_multiple * support_radius
            support_radius = support_radius.float().cuda(non_blocking=True)
            support_angle = (opt.support_angle / 360) * 2 * np.pi
            gt_normal=gt_normal.float().cuda()
            normal=normal.float().cuda()
            index=index.cuda()
            # print(index.shape)
            # exit(0)

            noise_patch = noise_patch.transpose(2, 1).contiguous()
            patch_normal=patch_normal.transpose(2,1).contiguous()

            x,n,w,topidx= denoisenet(noise_patch, patch_normal,index)
            # loss,loss1,loss2=comtrative_loss(x,n,gt_patch,gt_patch_normal,w,gt_normal,support_radius,support_angle,opt.repulsion_alpha)
            loss,loss1,loss2,loss3=compute_bilateral_loss(x,n,gt_patch,gt_patch_normal,w,support_radius,support_angle,opt.repulsion_alpha,topidx)
            loss.backward()
            optimizer.step()

            print('[%d: %d/%d] train loss: %f\n' % (epoch, batch_ind, num_batch, loss.item()))
            train_writer.add_scalar('loss', loss.data.item(), epoch * num_batch + batch_ind)

            train_writer.add_scalar('loss1', loss1.data.item(), epoch * num_batch + batch_ind)
            train_writer.add_scalar('loss2', loss2.data.item(), epoch * num_batch + batch_ind)
            train_writer.add_scalar('loss3', loss3.data.item(), epoch * num_batch + batch_ind)
        checpoint_state = {
            'epoch': epoch + 1,
            'state_dict': denoisenet.state_dict(),
            'optimizer': optimizer.state_dict()}

        if epoch == (opt.nepoch - 1):

            torch.save(checpoint_state, '%s/model_full_ae.pth' % opt.network_model_dir)

        if epoch % opt.model_interval == 0:

            torch.save(checpoint_state, '%s/model_full_ae_%d.pth' % (opt.network_model_dir, epoch))

if __name__ == '__main__':
    parameters = parse_arguments()
    parameters.trainset = './trainset'
    parameters.summary_train = './log'
    parameters.network_model_dir = './Models/'
    parameters.batchSize = 128
    parameters.lr = 1e-4
    parameters.workers = 4
    parameters.nepoch =50
    print(parameters)
    train(parameters)
