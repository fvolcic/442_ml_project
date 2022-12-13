
import torch
import numpy as np 
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(img):
  tmp = img.detach().cpu().numpy().transpose(1, 2, 0)
  if tmp.shape[2] == 1:
    tmp = tmp[:, :, 0]
  return (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
  
def show_result(G, x_, y_, num_epoch):
  predict_images = torch.round(G(x_) * 12)
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


def convert_to_one_hot(segmentation_images, num_classes):
  """
  Convert a segmentation image to one hot encoding

  @params
  segmentation_images: (batch_size, height, width) or (batch_size, 1, height, width)
  num_classes: number of classes

  @return
  one_hot_images: (batch_size, num_classes, height, width)
  """

  if len(segmentation_images.shape) == 3:
    segmentation_images = segmentation_images.unsqueeze(1)
  return torch.nn.functional.one_hot(segmentation_images.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def convert_to_segmentation(one_hot_images):
  """
  Convert a one hot encoding image to segmentation image

  @params
  one_hot_images: (batch_size, num_classes, height, width)

  @return
  segmentation_images: (batch_size, 1, height, width)
  """
  return torch.argmax(one_hot_images, dim=1)