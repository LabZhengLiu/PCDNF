import os
import torch
import time
import numpy as np
from NetworkN1 import PCPNet
from dataset923 import PointcloudPatchDataset,my_collate
from utils118 import parse_arguments

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def eval(opt):


    with open(os.path.join(opt.testset, 'test.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    if not os.path.exists(parameters.save_dir):
        os.makedirs(parameters.save_dir)
    for shape_id, shape_name in enumerate(shape_names):
        print(shape_name)
        original_noise_pts = np.load(os.path.join(opt.testset, shape_name + '.npy'))
        np.save(os.path.join(opt.save_dir, shape_name + '_pred_iter_0.npy'), original_noise_pts.astype('float32'))
        for eval_index in range(opt.eval_iter_nums):
            print(eval_index)
            test_dataset = PointcloudPatchDataset(
                root=opt.save_dir,
                shape_name=shape_name + '_pred_iter_' + str(eval_index),
                patch_radius=opt.patch_radius,
                train_state='evaluation',
                knn=True)
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=opt.batchSize,
                collate_fn=my_collate,
                num_workers=int(opt.workers))

            pointfilter_eval = PCPNet()
            model_filename = os.path.join(parameters.eval_dir, 'model_full_ae.pth')
            checkpoint = torch.load(model_filename)
            pointfilter_eval.load_state_dict(checkpoint['state_dict'])

            pointfilter_eval.cuda()
            pointfilter_eval.eval()

            patch_radius = test_dataset.patch_radius_absolute
            pred_pts = np.empty((0, 6), dtype='float32')
            # start = time.time()/
            for batch_ind, data_tuple in enumerate(test_dataloader):
                #normal [64,3]
                noise_patch, noise_inv, noise_point,patch_normal,index,normals= data_tuple

                noise_patch = noise_patch.float().cuda()
                noise_inv = noise_inv.float().cuda()
                patch_normal=patch_normal.float().cuda()
                index=index.cuda()
                normals=normals.float().cuda()

                noise_patch = noise_patch.transpose(2, 1).contiguous()
                patch_normal=patch_normal.transpose(2,1).contiguous()

                with torch.no_grad():
                    #dis= pointfilter_eval(noise_patch,patch_normal)  # [64,3]
                    dis,n,_,_= pointfilter_eval(noise_patch, patch_normal,index)
                    # dis,classficaton,pointfval = pointfilter_eval(noise_patch,distance)#[64,3]
                dis=dis.unsqueeze(2)
                # n=n[:,0,:]
                n=n.unsqueeze(2)

                dis = torch.bmm(noise_inv, dis)#[64,3,1]
                n=torch.bmm(noise_inv,n)
                dis=np.squeeze(dis.data.cpu().numpy()) * patch_radius + noise_point.numpy()
                n=np.squeeze(n.data.cpu().numpy())
                normal=n
                #normal=normal.data.cpu().numpy()
                # print(dis.shape)
                # print(normal.shape)
                if normal.shape[0] != dis.shape[0]:
                    normal = normal.reshape(dis.shape)
                # exit(0)
                pred_normal=np.append(dis,normal,axis=1)
                pred_pts = np.append(pred_pts,
                                    pred_normal,axis=0)
            end = time.time()
            print("total_time:"+str(end-start))
            np.save(os.path.join(opt.save_dir, shape_name + '_pred_iter_' + str(eval_index + 1) + '.npy'),
                    pred_pts.astype('float32'))
            np.save(os.path.join(opt.save_dir, shape_name + '_pred_iter_' + str(eval_index + 1) + '.npy'),
                    pred_pts.astype('float32'))
            # np.savetxt(os.path.join(opt.save_dir, shape_name + '.txt'),
            #         pred_pts.astype('float32'), fmt='%.6f')



if __name__ == '__main__':

    parameters = parse_arguments()
    parameters.testset = r'testdir'
    parameters.eval_dir = './Trained_Models/'
    parameters.batchSize = 64
    parameters.eval_iter_nums =1
    parameters.workers = 4
    parameters.save_dir = r'savedir'
    parameters.patch_radius = 0.05
    eval(parameters)