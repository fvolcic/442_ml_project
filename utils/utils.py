
import torch
import numpy as np 
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(img):
    tmp = img
    if tmp.shape[0] == 13:
        tmp = convert_to_segmentation(tmp).unsqueeze(0)
    elif tmp.shape[0] == 4:
        tmp = tmp[:3,:,:]
    tmp = tmp.cpu().numpy().transpose(1, 2, 0)
    return tmp
  
def show_result(G, x_, y_, num_epoch):
    predict_images = G(x_)
    fig, ax = plt.subplots(x_.size()[0], 3, figsize=(6,10))
    for i in range(x_.size()[0]):
        ax[i, 0].get_xaxis().set_visible(False)
        ax[i, 0].get_yaxis().set_visible(False)
        ax[i, 1].get_xaxis().set_visible(False)
        ax[i, 1].get_yaxis().set_visible(False)
        ax[i, 2].get_xaxis().set_visible(False)
        ax[i, 2].get_yaxis().set_visible(False)
        ax[i, 0].cla()
        ax[i, 0].imshow(process_image(x_[i]))
        ax[i, 1].cla()
        ax[i, 1].imshow(process_image(predict_images[i]))
        ax[i, 2].cla()
        ax[i, 2].imshow(process_image(torch.round(y_[i] * 12)))
  
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


def convert_to_one_hot(labels, classes):
  """
  Convert a segmentation image to one hot encoding

  @params
  segmentation_images: (batch_size, height, width) or (batch_size, 1, height, width)
  num_classes: number of classes

  @return
  one_hot_images: (batch_size, num_classes, height, width)
  """

  if len(labels.shape) == 4:
    labels = labels.squeeze(1)

  labels = torch.round(labels * 12)
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