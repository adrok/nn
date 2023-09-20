from torch import nn
import torch
import torch.nn.functional as F


# https://www.kaggle.com/code/faizalkarim/image-colorization-auto-encoders-pytorch
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.t_conv4 = nn.ConvTranspose2d(192, 15, 3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.converge = nn.Conv2d(18, 3, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        xd = F.relu(self.t_conv1(x4))
        xd = torch.cat((xd, x3), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv2(xd))
        xd = torch.cat((xd, x2), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv3(xd))
        xd = torch.cat((xd, x1), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv4(xd))
        xd = torch.cat((xd, x), dim=1)
        x_out = F.relu(self.converge(xd))
        return x_out

    # def __init__(self):
    #     super().__init__()
    #
    #     self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
    #     # self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
    #     # self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #
    #     # decoder layers
    #     self.t_conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
    #     self.t_conv2 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)
    #     # self.t_conv3 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
    #     # self.t_conv4 = nn.ConvTranspose2d(192, 15, 3, stride=1, padding=1)
    #     # self.dropout = nn.Dropout(0.2)
    #     # self.converge = nn.Conv2d(18, 3, 3, stride=1, padding=1)
    #
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = self.pool(x)
    #     x = F.relu(self.t_conv1(x))
    #     x = F.relu(self.t_conv2(x))
    #     # x3 = F.relu(self.conv3(x2))
    #     # x4 = F.relu(self.conv4(x3))
    #     #
    #     # xd = F.relu(self.t_conv1(x4))
    #     # xd = torch.cat((xd, x3), dim=1)
    #     # xd = self.dropout(xd)
    #     # xd = F.relu(self.t_conv2(xd))
    #     # xd = torch.cat((xd, x2), dim=1)
    #     # xd = self.dropout(xd)
    #     # xd = F.relu(self.t_conv3(xd))
    #     # xd = torch.cat((xd, x1), dim=1)
    #     # xd = self.dropout(xd)
    #     # xd = F.relu(self.t_conv4(xd))
    #     # xd = torch.cat((xd, x), dim=1)
    #     # x_out = F.relu(self.converge(xd))
    #     return x
