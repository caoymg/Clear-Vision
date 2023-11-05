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
from data_loader import EarthNet2021Dataset
from Networks import *
import torchvision.models as models
from PIL import Image


if torch.cuda.is_available():
    cuda = 1
    print("CUDA is available on this system.")
else:
    cuda = 0
    print("CUDA is not available on this system.")
    
    
def metric(prediction, ground_truth):
    
    metric_dict = {}

    # Calculate the element-wise squared differences and
    # the mean of the squared differences along the appropriate dimension(s)
    mse = torch.mean(torch.square(prediction - ground_truth))
    # Take the square root of the mean to get the RMSE
    rmse = torch.sqrt(mse)
    metric_dict["rmse"] = rmse
    
    # Compute the maximum possible pixel value for the data type (e.g., 255 for uint8)
    peak_value = torch.max(ground_truth)
    # Calculate the PSNR using the formula
    psnr = 20 * torch.log10(peak_value) - 10 * torch.log10(mse)
    metric_dict["psnr"] = psnr
    
    return metric_dict


def visualize(masked_data, T, gt_data, pred, iter_num):
    """
    image shape: bs, 10, 3, 128, 128 
    """
    
    
    for idx in range(T):
                            
        r = masked_data[0,idx,2,:,:]
        g = masked_data[0,idx,1,:,:]
        b = masked_data[0,idx,0,:,:]
        rgb_image = np.dstack((r,g,b))
        pil_image = Image.fromarray((rgb_image*255).astype('uint8'))
        pil_image.save('/home/tmp_img/'+str(iter_num)+'_masked_data_idx'+str(idx)+'_.png')

        r = gt_data[0,idx,2,:,:]
        g = gt_data[0,idx,1,:,:]
        b = gt_data[0,idx,0,:,:]
        rgb_image = np.dstack((r,g,b))
        pil_image = Image.fromarray((rgb_image*255).astype('uint8'))
        pil_image.save('/home/tmp_img/'+str(iter_num)+'_gt_data_idx'+str(idx)+'_.png')    

        r = pred[0,idx,2,:,:]
        g = pred[0,idx,1,:,:]
        b = pred[0,idx,0,:,:]
        rgb_image = np.dstack((r,g,b))
        pil_image = Image.fromarray((rgb_image*255).astype('uint8'))
        pil_image.save('/home/tmp_img/'+str(iter_num)+'_pred_idx'+str(idx)+'_.png')    


def evaluate(iter_num, model, test_loader, max_psnr, args):
    # print("TESTING...")
    rmse_lst = []
    psnr_lst = []
    model.eval()
    for batch_index, test_data in enumerate(test_loader):
        # if batch_index == 1:
        #     break
        batch_images = test_data["images"]                                         # batch_images: bs, 10, 3, 128, 128 
        reshaped_images = batch_images.transpose(1,2)                               # Reshape to [bs, 3(C), 10(T), 128, 128]
        batch_gt = test_data["gt"]
        batch_cubenames = test_data["cubename"]
        
        if cuda:
            reshaped_images = reshaped_images.cuda()
            batch_gt = batch_gt.cuda()
        _, pred = model(reshaped_images)                                      # pred [bs, 3, 10, 128, 128]
        pred = pred.transpose(1,2)
        # mask the padded frames
        non_padded_mask = batch_gt != 255*torch.ones_like(batch_gt)
        valid_pred = pred[non_padded_mask].detach()
        valid_gt = batch_gt[non_padded_mask].detach()
        if batch_index == 0:
            valid_pred_all = valid_pred
            valid_gt_all = valid_gt
        else:
            valid_pred_all = torch.cat((valid_pred, valid_pred_all), 0)
            valid_gt_all = torch.cat((valid_gt, valid_gt_all), 0)

    metric_dict = metric(valid_pred_all.detach(), valid_gt_all.detach())
    print("> TEST rmse = %f, psnr  = %f" % (metric_dict["rmse"], metric_dict["psnr"]))
    if metric_dict["psnr"] > max_psnr:
        max_psnr = metric_dict["psnr"]
        model_name = 'iter_num_'+repr(iter_num)+'_psnr_'+repr(metric_dict["psnr"].item())+'.pth'
        torch.save(model.state_dict(), os.path.join(args.snapshot_dir, model_name))
        print("*** achieve better psnr on testset. save state dict.")
    
    return max_psnr, metric_dict
            
            
def main(args):


    model = unet()

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
                    EarthNet2021Dataset(args.root_dir+"/processed_train"),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    test_loader = data.DataLoader(
                    EarthNet2021Dataset(args.root_dir+"/processed_test"),
                    batch_size=args.test_batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(),
                        lr=args.learning_rate, weight_decay=args.weight_decay)

    
    l1loss = nn.L1Loss()
    
    
    min_loss = 0.09
    
    max_psnr, metric_dict = evaluate(0, model, test_loader, 0, args)
    
    loss_lst = []

    for iter_num in range(args.num_epoch):
        iter_loss_lst = []
        for batch_index, train_data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()       
            batch_images = train_data["images"]                                         # batch_images: bs, 10, 3, 128, 128 
            reshaped_images = batch_images.transpose(1,2)                               # Reshape to [bs, 3(C), 10(T), 128, 128]
            batch_gt = train_data["gt"]
            batch_cubenames = train_data["cubename"]
            
            if cuda:
                reshaped_images = reshaped_images.cuda()
                batch_gt = batch_gt.cuda()
            _, pred = model(reshaped_images)                                            # pred [bs, 3, 10, 128, 128]
            pred = pred.transpose(1,2)
            pred_copy = pred
            # mask the padded frames
            non_padded_mask = batch_gt != 255*torch.ones_like(batch_gt)
            # print("Number of real value elements:", non_padded_mask.sum().item())
            
            loss = l1loss(pred[non_padded_mask], batch_gt[non_padded_mask])
            loss.backward()
            optimizer.step()
            
            metric_dict = metric(pred[non_padded_mask].detach(), batch_gt[non_padded_mask].detach())
            
            if batch_index%100 == 0:
                print("Train iter %d, batch_index = %d/%d, loss = %f, rmse = %f, psnr  = %f" % (iter_num, batch_index,len(train_loader), loss, metric_dict["rmse"], metric_dict["psnr"]))
            
            # For each decent loss, perform test eval and visualization.
            if loss < min_loss:
                min_loss = loss
                max_psnr, metric_dict = evaluate(iter_num, model, test_loader, max_psnr, args)
                visualize(batch_images.detach().cpu().numpy(), 10, train_data["gt"].detach().cpu().numpy(), pred_copy.detach().cpu().numpy(), iter_num)
                print("achieve smaller loss on trainset. finished vis with loss", loss)

            iter_loss_lst.append(loss)
        # For each ITER, perform test eval.
        max_psnr, metric_dict = evaluate(iter_num, model, test_loader, max_psnr, args)
        mean_loss = sum(iter_loss_lst)/len(iter_loss_lst)
        loss_lst.append(mean_loss)
        print("iter_loss_lst", iter_loss_lst)
        print("loss_lst", loss_lst)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--root_dir', type=str, default='/datadrive/model_and_dict',help='dataset path.')    
    parser.add_argument('--ignore_label', type=int, default=0,
                        help='the index of the label ignored in the training.')     
    parser.add_argument('--batch_size', type=int, default= 3,
                        help='number of images in each batch.')
    parser.add_argument('--test_batch_size', type=int, default= 1,
                        help='number of images in each batch.')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='base learning rate.')
    parser.add_argument('--num_epoch', type=int, default=1000,
                        help='Number of training steps.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='regularisation parameter for L2-loss.')
    parser.add_argument('--snapshot_dir', type=str, default='./model_dict/',
                        help='where to save snapshots of the model.')
    
    torch.cuda.manual_seed_all(0)

    main(parser.parse_args())
