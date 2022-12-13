from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model
from keras.applications import ResNet50
import torch
import torch.nn as nn

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resnet50_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    # resnet50.summary()
    # resnet50 = keras.applications.ResNet50(weights=weights_path ,include_top=False, input_shape=(n_h, n_w, n_c))
    # resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)        
    """ Encoder """
    s1 = resnet50.layers[0].output           ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model

class resNet(nn.Module):
    def __init__(self, input_shape):
        self.model = build_resnet50_unet(input_shape)
        
    def forward(self, x):
        return self.model(x)
    
# import torch
# import torch.nn as nn
# import torchvision
# resnet = torchvision.models.resnet.resnet50(pretrained=True)


# class ConvBlock(nn.Module):
#     """
#     Helper module that consists of a Conv -> BN -> ReLU
#     """

#     def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.with_nonlinearity = with_nonlinearity

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.with_nonlinearity:
#             x = self.relu(x)
#         return x


# class Bridge(nn.Module):
#     """
#     This is the middle layer of the UNet which just consists of some
#     """

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.bridge = nn.Sequential(
#             ConvBlock(in_channels, out_channels),
#             ConvBlock(out_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.bridge(x)


# class UpBlockForUNetWithResNet50(nn.Module):
#     """
#     Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
#     """

#     def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
#                  upsampling_method="conv_transpose"):
#         super().__init__()

#         if up_conv_in_channels == None:
#             up_conv_in_channels = in_channels
#         if up_conv_out_channels == None:
#             up_conv_out_channels = out_channels

#         if upsampling_method == "conv_transpose":
#             self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
#         elif upsampling_method == "bilinear":
#             self.upsample = nn.Sequential(
#                 nn.Upsample(mode='bilinear', scale_factor=2),
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#             )
#         self.conv_block_1 = ConvBlock(in_channels, out_channels)
#         self.conv_block_2 = ConvBlock(out_channels, out_channels)

#     def forward(self, up_x, down_x):
#         """
#         :param up_x: this is the output from the previous up block
#         :param down_x: this is the output from the down block
#         :return: upsampled feature map
#         """
#         x = self.upsample(up_x)
#         x = torch.cat([x, down_x], 1)
#         x = self.conv_block_1(x)
#         x = self.conv_block_2(x)
#         return x


# class UNetWithResnet50Encoder(nn.Module):
#     DEPTH = 4

#     def __init__(self, n_classes=1):
#         super().__init__()
#         resnet = torchvision.models.resnet.resnet50(pretrained=True)
#         down_blocks = []
#         up_blocks = []
#         self.input_block = nn.Sequential(*list(resnet.children()))[:3]
#         self.input_pool = list(resnet.children())[3]
#         for bottleneck in list(resnet.children()):
#             if isinstance(bottleneck, nn.Sequential):
#                 down_blocks.append(bottleneck)
#         self.down_blocks = nn.ModuleList(down_blocks)
#         self.bridge = Bridge(2048, 2048)
#         up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
#         up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
#         up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
#         up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
#                                                     up_conv_in_channels=256, up_conv_out_channels=128))
#         up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
#                                                     up_conv_in_channels=128, up_conv_out_channels=64))

#         self.up_blocks = nn.ModuleList(up_blocks)

#         self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

#     def forward(self, x, with_output_feature_map=False):
#         pre_pools = dict()
#         pre_pools[f"layer_0"] = x
#         x = self.input_block(x)
#         pre_pools[f"layer_1"] = x
#         x = self.input_pool(x)

#         for i, block in enumerate(self.down_blocks, 2):
#             x = block(x)
#             if i == (UNetWithResnet50Encoder.DEPTH - 1):
#                 continue
#             pre_pools[f"layer_{i}"] = x

#         x = self.bridge(x)

#         for i, block in enumerate(self.up_blocks, 1):
#             key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
#             x = block(x, pre_pools[key])
#         output_feature_map = x
#         x = self.out(x)
#         del pre_pools
#         if with_output_feature_map:
#             return x, output_feature_map
#         else:
#             return x


