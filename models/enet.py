import torch.nn as nn
import torch

class ENET(nn.module):

    class FirstBlock(nn.module):
        def __init__(self, in_channels=3, out_channels=13):
            self.3x3 = nn.conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride=2, padding=1)
            self.max_pool = nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0)
            self.PReLU = nn.PReLU(num_features = 16)
            self.batchNorm2D = nn.batchNorm2d(num_features = 16)
            

        def forward(self, input):
            three_X_three = self.3x3(input)
            max_pool = self.max_pool(input)
            result = torch.cat((three_X_three, max_pool), 1)
            result = self.batchNorm2D(result)
            result = self.PReLU(result)
            return result
    
    class StageOneBottleNeck(nn.module):
        def __init__(self, in_channels, out_channels):
            bneck_channels = in_channels//4
            self.right_branch = nn.sequential(
                nn.conv2d(in_channels, bneck_channels, kernel_size = 2, bias = False),
                nn.batchNorm2d(bneck_channels),
                nn.PReLU(),
                nn.conv2d(bneck_channels, bneck_channels, kernel_size = 3, stride = 2, padding = 1, bias = False),
                nn.batchNorm2d(bneck_channels),
                nn.PReLU(),
                nn.conv2d(bneck_channels, out_channels, kernel_size = 1, bias = False),
                nn.batchNorm2d(out_channels),
                nn.PReLU(),
                nn.Dropout2d(p=0.01),
                nn.PReLU()
            )

            self.padding = out_channels - in_channels

            self.maxPool = nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0, return_indices = True)

        def forward(self, input):
            right_result = self.right_branch(input)
            left_result, indices = self.maxPool(input)

            if self.pad > 0:
                pad = torch.zeros((input.size(0), self.padding, input.size(2), input.size(3))).to(device=device)
                input = input.cat((input,pad), 1)
