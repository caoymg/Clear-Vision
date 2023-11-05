from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import os
import time

def generate_cloud_mask(cuda, model, data_loader):

    model.eval()
    for batch_index, test_data in enumerate(data_loader):

        batch_images = test_data["images"]                                         # batch_images: bs, 3, 128, 128 

        if cuda:
            batch_images = batch_images.cuda()

        _, pred = model(batch_images)                                              # pred [bs, 2, 128, 128]
        _, batch_cloud_masks = torch.max(nn.functional.softmax(pred,dim=1).detach(), 1)

        batch_cloud_masks = batch_cloud_masks.detach()
        
        if batch_index == 0:
            cloud_masks = batch_cloud_masks
            images = batch_images
        else:
            cloud_masks = torch.cat((cloud_masks, batch_cloud_masks), 0)
            images = torch.cat((images, batch_images), 0)
    
    return images, cloud_masks


def cloud_removal(cuda, model, images):
        
    model.eval()
    
    images = images.transpose(1,2)                                                  # Reshape to [1, 3(C), 10(T), 128, 128]

    if cuda:
        images = images.cuda()

    _, pred_images = model(images)                                                  # pred [bs, 3, 10, 128, 128]
    pred_images = pred_images.transpose(1,2)                                        # pred_images [bs, 10, 3, 128, 128]
    
    return pred_images
    
