import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch.optim as optim
import time
from tqdm import tqdm
from utils.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNetAI(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetAI, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.final = DoubleConv(n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
def train_UNet_AI(model, train_loader, num_epochs):
    tmp = next(iter(train_loader))
    fixed_x = tmp[0].to(device=device)
    fixed_y = tmp[1].to(device=device)
    fixed_x_D = torch.cat([tmp[0].to(device=device), tmp[3].to(device=device)],1)
    fixed_y_D = tmp[1].to(device=device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(num_epochs):
        losses = []
        epoch_start_time = time.time()
        batch_idx = 0
        for data in tqdm(train_loader):
            input = data[0].to(device=device) # images
            
            if (model.n_channels == 4):
                depth = data[3].to(device=device)
                input = torch.cat([input, depth], 1)
            
            target = data[1].to(device=device) # sementic segmentations
            target = convert_to_one_hot(torch.round(target * 12), model.n_classes)
            model.zero_grad()
            optimizer.zero_grad()
            output = model(input)
            loss = F.binary_cross_entropy_with_logits(output, target)
            
            loss.backward()
            optimizer.step()
            
            _loss = loss.detach().item()
            losses.append(_loss)
            batch_idx += 1
            
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - using time: %.2f seconds' % ((epoch + 1), num_epochs, per_epoch_ptime))
        print('loss of generator G: %.3f' % (torch.mean(torch.FloatTensor(losses))))
        if epoch == 0 or (epoch + 1) % 5 == 0:
            with torch.no_grad():
                if (model.n_channels == 4):
                    show_result(model, fixed_x_D, fixed_y_D, (epoch+1))
                else:
                    show_result(model, fixed_x, fixed_y, (epoch+1))
        
def test_UNet_AI(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.binary_cross_entropy_with_logits(output, target, size_average=False).item() # sum up batch loss
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))