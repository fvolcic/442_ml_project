
import torch
import numpy as np 
import matplotlib.pyplot as plt
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(img):
    tmp = img
    if tmp.shape[1] == 13:
        tmp = convert_to_segmentation(tmp)
    elif tmp.shape[1] == 4 or tmp.shape[1] == 1:
        tmp = tmp[:,:3,:,:].permute(0,2,3,1)
    return tmp.cpu()
  
def show_result(G, x_, y_, num_epoch):
    preds = process_image(G(x_))
    x = process_image(x_)
    y = process_image(y_)
    fig, ax = plt.subplots(x_.shape[0], 3, figsize=(6,10))
    for i in range(x_.shape[0]):
        ax[i, 0].get_xaxis().set_visible(False)
        ax[i, 0].get_yaxis().set_visible(False)
        ax[i, 1].get_xaxis().set_visible(False)
        ax[i, 1].get_yaxis().set_visible(False)
        ax[i, 2].get_xaxis().set_visible(False)
        ax[i, 2].get_yaxis().set_visible(False)
        ax[i, 0].cla()
        ax[i, 0].imshow(x[i])
        ax[i, 1].cla()
        ax[i, 1].imshow(preds[i])
        ax[i, 2].cla()
        ax[i, 2].imshow(torch.round(y[i] * 12))
  
    plt.tight_layout()
    label_epoch = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0, label_epoch, ha='center')
    label_input = 'Input'
    fig.text(0.18, 1, label_input, ha='center')
    label_output = 'Output'
    fig.text(0.5, 1, label_output, ha='center')
    label_truth = 'Ground truth'
    fig.text(0.81, 1, label_truth, ha='center')

    plt.show()

def convert_to_one_hot(labels, classes=13, scale_values=True):
    """
    Convert a segmentation image to one hot encoding

    @params
    labels: (batch_size, height, width) or (batch_size, 1, height, width)
    classes: number of classes
    scale_values: if True, labels is multiplied by classes then rounded to the nearest integer before converting to one hot encoding.

    @note
    labels must be integers between 0 and classes-1 unless scale_values is True. 
    If scale_values is True, labels must be between 0 and 1, 
    and will be multiplied by classes before being converted to one hot encoding.

    @return
    one_hot_images: (batch_size, num_classes, height, width)
    """

    if len(labels.shape) == 4:
        labels = labels.squeeze(1)
    
    if scale_values:
        labels = torch.round(labels * (classes-1))
    
    one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=classes)
    one_hot = torch.permute(one_hot, [0, 3, 1, 2]).float()

    return one_hot

def convert_to_segmentation(one_hot_images):
    """
    Convert a one hot encoding image to segmentation image

    @params
    one_hot_images: (batch_size, num_classes, height, width)

    @return
    segmentation_images: (batch_size, 1, height, width)
    """
    return torch.argmax(one_hot_images, dim=1)

def meanIoU(pred, target, classes):
    """
    Give the mean IoU of the image and the mask
    
    @params
        img: (batch_size, classes, height, width)
        mask: (batch_size, classes, height, width)
        classes: number of classes
        
    @note it is required that the values in img and mask and intergers between 0 and classes-1 

    @returns
        mean IoU
    """
    
    # convert pred to segmentation and back to one hot encoding
    pred = convert_to_one_hot(convert_to_segmentation(pred), classes=classes, scale_values=False)
    
    intersection = torch.sum(pred * target, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    iou = torch.mean(iou, dim=1)
    return torch.mean(iou)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        intersection = torch.sum(pred * target, dim=(1, 2, 3))
        union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
        dice = (2 * intersection ) / (union + 1e-8)
        dice = torch.mean(1 - dice)
        return dice 