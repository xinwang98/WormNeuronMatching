import torch.nn as nn
import torch

class ClassifierNetwork(nn.Module):
    def __init__(self, bin_num, edge_num, output_planes=107):
        super(ClassifierNetwork, self).__init__()
        self.patch_feature = nn.Sequential(
            nn.Conv2d(23, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.vertical_feature = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.linear = nn.Sequential(
            nn.Linear(4*4*512+2*2*512, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),
            nn.Dropout(),

            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(),

            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(),

            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Dropout(),

            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Dropout()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256+3*bin_num+edge_num*5,512),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(),

            nn.Linear(512,256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(128, output_planes)
        )

    def forward(self, patch, vertical_patch, pair):

        out_patch = self.patch_feature(patch)
        out_patch = out_patch.view(out_patch.shape[0], -1)
        out_vertical_patch =self.vertical_feature(vertical_patch)
        out_vertical_patch = out_vertical_patch.view(vertical_patch.shape[0], -1)

        out = self.linear(torch.cat((out_patch,out_vertical_patch),dim=1))
        out = torch.cat((out, pair),dim=1)
        out = self.classifier(out)
        return out