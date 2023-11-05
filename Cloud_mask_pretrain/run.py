import argparse
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torch.backends.cudnn as cudnn
from data_loader import CloudMaskDataset
from Networks import *
import torchvision.models as models
from PIL import Image


if torch.cuda.is_available():
    cuda = 1
    print("CUDA is available on this system.")
else:
    cuda = 0
    print("CUDA is not available on this system.")
    
def visualize(batch_images, batch_gt, pred):
    
    """
    batch_images: bs, 3, 128, 128 
    batch_gt, pred: bs, 128, 128 
    """

    batch_images = batch_images.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    batch_gt = batch_gt.detach().cpu().numpy()

    for idx in range(pred.shape[0]):
        
        if idx == 10:
            break
        
        r = batch_images[idx,2,:,:]
        g = batch_images[idx,1,:,:]
        b = batch_images[idx,0,:,:]
        rgb_image = np.dstack((r,g,b))
        pil_image = Image.fromarray((rgb_image*255).astype('uint8'))
        pil_image.save('/home/tmp_img/cloud/'+'original_image'+str(idx)+'.png')

        # mask the cloud with max intensity 1
        rgb_image[:,:,0][pred[idx] == 1] = 1
        rgb_image[:,:,1][pred[idx] == 1] = 1
        rgb_image[:,:,2][pred[idx] == 1] = 1
        pil_image = Image.fromarray((rgb_image*255).astype('uint8'))
        pil_image.save('/home/tmp_img/cloud/'+'pred_image'+str(idx)+'.png')

        rgb_image = np.dstack((r,g,b))
        rgb_image[:,:,0][batch_gt[idx] == 1] = 1
        rgb_image[:,:,1][batch_gt[idx] == 1] = 1
        rgb_image[:,:,2][batch_gt[idx] == 1] = 1
        pil_image = Image.fromarray((rgb_image*255).astype('uint8'))
        pil_image.save('/home/tmp_img/cloud/'+'target_image'+str(idx)+'.png')


def metric(predict, label, n_classes, test_flg):

    epsilon = 1e-14
    name_classes = ["non-cloud", "cloud"]
    
    TP = np.zeros((n_classes, 1))
    FP = np.zeros((n_classes, 1))
    TN = np.zeros((n_classes, 1))
    FN = np.zeros((n_classes, 1))
    F1 = np.zeros((n_classes, 1))
    
    for i in range(0, n_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)        

    OA = np.sum(TP)*1.0 / label.size
    for i in range(n_classes):
        P = TP[i]*1.0 / (TP[i] + FP[i] + epsilon)
        R = TP[i]*1.0 / (TP[i] + FN[i] + epsilon)
        F1[i] = 2.0*P*R / (P + R + epsilon)
    
    if test_flg:
        for i in range(n_classes):
            print('==> F1 ' + name_classes[i] + ': %.2f'%(F1[i] * 100))

    mF1 = np.mean(F1)    

    return mF1, OA


def evaluate(iter_num, model, test_loader, max_mF1, n_classes, args):

    model.eval()
    for batch_index, test_data in enumerate(test_loader):

        batch_images = test_data["images"]                                         # batch_images: bs, 3, 128, 128 
        batch_gt = test_data["gt"]

        if cuda:
            batch_images = batch_images.cuda()
            batch_gt = batch_gt.cuda()

        _, pred = model(batch_images)                                              # pred [bs, 2, 128, 128]
        _,pred = torch.max(nn.functional.softmax(pred,dim=1).detach(), 1)

        valid_pred = pred.detach()
        valid_gt = batch_gt.detach()
        
        if batch_index == 0:
            valid_pred_all = valid_pred
            valid_gt_all = valid_gt
        else:
            valid_pred_all = torch.cat((valid_pred, valid_pred_all), 0)
            valid_gt_all = torch.cat((valid_gt, valid_gt_all), 0)

    test_flg = 1
    mF1, OA = metric(valid_pred_all.detach().cpu().numpy(), valid_gt_all.detach().cpu().numpy(), n_classes, test_flg)
    print(">> TEST mF1 = %f, OA  = %f" % (mF1, OA))
    
    if mF1 > max_mF1:
        max_mF1 = mF1
        model_name = 'iter_num_'+repr(iter_num)+'_mF1_'+repr(mF1.item())+'.pth'
        torch.save(model.state_dict(), os.path.join(args.snapshot_dir, model_name))
        print("*** achieve better mF1 on testset. save state dict.")
    
    visualize(batch_images, batch_gt, pred)

    return max_mF1
        
        
def main(args):

    n_classes = 2
    model = unet(n_classes)

    model_path = os.listdir(args.snapshot_dir)
    for filename in model_path: 
        filepath = os.path.join(args.snapshot_dir, filename)    
        print("filename", filename)       
        saved_state_dict = torch.load(filepath)
        model.load_state_dict(saved_state_dict)   
        print("*** load state dict")
    
    if cuda:
        model = model.cuda()
        
    train_loader = data.DataLoader(
                    CloudMaskDataset(args.root_dir+"/real_test_iid", "train"),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    test_loader = data.DataLoader(
                    CloudMaskDataset(args.root_dir+"/real_test_ood", "test"),
                    batch_size=args.test_batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(),
                        lr=args.learning_rate, weight_decay=args.weight_decay)
    
    seg_loss = nn.CrossEntropyLoss()
    
    
    min_loss = 0.1
    
    max_mF1 = evaluate(0, model, test_loader, 0, n_classes, args)
    
    

    for iter_num in range(args.num_epoch):
        iter_loss_lst = []
        for batch_index, train_data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()       
            batch_images = train_data["images"]                                         # batch_images: bs, 3, 128, 128 
            batch_gt = train_data["gt"]                                                 # batch_images: bs, 128, 128 
            
            if cuda:
                batch_images = batch_images.cuda()
                batch_gt = batch_gt.cuda()
        
            _, pred = model(batch_images)                                            # pred [bs, 2, 128, 128]
            loss = seg_loss(pred, batch_gt.long())
            loss.backward()
            optimizer.step()
            
            test_flg = 0
            _,pred = torch.max(nn.functional.softmax(pred,dim=1).detach(), 1)
            mF1, OA = metric(pred.detach().cpu().numpy(), batch_gt.detach().cpu().numpy(), n_classes, test_flg)

            if batch_index%10 == 0:
                print("Train iter %d, batch_index = %d/%d, loss = %f, mF1 = %f, OA = %f" % (iter_num, batch_index,len(train_loader), loss, mF1, OA))
            
                # For each decent loss, perform test eval and visualization.
                if loss < min_loss:
                    min_loss = loss
                    max_mF1 = evaluate(iter_num, model, test_loader, max_mF1, n_classes, args)
            
        # For each ITER, perform test eval.
        max_mF1 = evaluate(iter_num, model, test_loader, max_mF1, n_classes, args)

        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--root_dir', type=str, default='/datadrive',help='dataset path.')    
    parser.add_argument('--ignore_label', type=int, default=0,
                        help='the index of the label ignored in the training.')     
    parser.add_argument('--batch_size', type=int, default= 8,
                        help='number of images in each batch.')
    parser.add_argument('--test_batch_size', type=int, default= 32,
                        help='number of images in each batch.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='base learning rate.')
    parser.add_argument('--num_epoch', type=int, default=1000,
                        help='Number of training steps.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='regularisation parameter for L2-loss.')
    parser.add_argument('--snapshot_dir', type=str, default='./model_dict/',
                        help='where to save snapshots of the model.')
    
    torch.cuda.manual_seed_all(0)

    main(parser.parse_args())
