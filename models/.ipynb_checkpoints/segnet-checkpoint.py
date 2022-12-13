
import torch
import torch.nn as nn 

# a resnet block with a bottleneck

class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block()
    
    def build_conv_block(self):
        conv_block = []
        conv_block += [nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(512)]
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out = self.conv_block(x)
        out = out + x
        return out
    
# the encoder part of the segnet

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_block1 = self.build_conv_block(3, 64)
        self.conv_block2 = self.build_conv_block(64, 128)
        self.conv_block3 = self.build_conv_block(128, 256)
        self.conv_block4 = self.build_conv_block(256, 512)
        self.conv_block5 = self.build_conv_block(512, 512)
        self.resnet_block = ResnetBlock()
    
    def build_conv_block(self, in_channels, out_channels):
        conv_block = []
        conv_block += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        out4 = self.conv_block4(out3)
        out5 = self.conv_block5(out4)
        out6 = self.resnet_block(out5)
        return out1, out2, out3, out4, out5, out6
    
# the decoder part of the segnet

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_block1 = self.build_conv_block(512, 512)
        self.conv_block2 = self.build_conv_block(512, 256)
        self.conv_block3 = self.build_conv_block(256, 128)
        self.conv_block4 = self.build_conv_block(128, 64)
        self.conv_block5 = self.build_conv_block(64, 64)
        self.conv_block6 = self.build_conv_block(64, 12)
    
    def build_conv_block(self, in_channels, out_channels):
        conv_block = []
        conv_block += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        out4 = self.conv_block4(out3)
        out5 = self.conv_block5(out4)
        out6 = self.conv_block6(out5)
        return out6

# the segnet model

class Segnet(nn.Module):
    def __init__(self):
        super(Segnet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        out1, out2, out3, out4, out5, out6 = self.encoder(x)
        out7 = self.decoder(out6)
        return out7
    
# loss function

class SegnetLoss(nn.Module):
    def __init__(self):
        super(SegnetLoss, self).__init__()
    
    def forward(self, pred, target):
        loss = nn.CrossEntropyLoss()
        return loss(pred, target)
    
