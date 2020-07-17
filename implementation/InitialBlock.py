import torch
import torch.nn as nn

class InitialBlock(nn.Module):

    def __init__ (self,in_channels = 3,out_channels = 13):
        super().__init__()


        self.maxpool = nn.MaxPool2d(kernel_size=2, 
                                      stride = 2, 
                                      padding = 0)

        self.conv = nn.Conv2d(in_channels, 
                                out_channels,
                                kernel_size = 3,
                                stride = 2, 
                                padding = 1)

        self.prelu = nn.PReLU(16)

        self.batchnorm = nn.BatchNorm2d(out_channels)
  
    def forward(self, input_image):
        
        conv_out = self.conv(input_image)
        conv_out = self.batchnorm(conv_out)
        
        max_out = self.maxpool(input_image)
        
        x = torch.cat((conv_out, max_out), dim=1)
        initialblock_out = self.prelu(x)
        
        return initialblock_out