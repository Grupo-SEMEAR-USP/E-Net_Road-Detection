import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UpsamplingBottleneck(nn.Module):

    def __init__ (self, in_channels, out_channels, projection_ratio_conv=4, relu = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        
        self.reduced_channels = int(in_channels//projection_ratio_conv)
        
        self.unpooling = nn.MaxUnpool2d(kernel_size = 2,
                                     stride = 2)
        
        self.main_conv = nn.Conv2d(in_channels = self.in_channels,
                                    out_channels = self.out_channels,
                                    kernel_size = 1)
        
        self.dropout = nn.Dropout2d(p=0.1)
        
        self.convt1 = nn.ConvTranspose2d(in_channels = self.in_channels,
                               out_channels = self.reduced_channels,
                               kernel_size = 1,
                               padding = 0,
                               bias = False)
        
        
        self.prelu1 = activation
        
        self.convt2 = nn.ConvTranspose2d(in_channels = self.reduced_channels,
                                  out_channels = self.reduced_channels,
                                  kernel_size = 3,
                                  stride = 2,
                                  padding = 1,
                                  output_padding = 1,
                                  bias = False)
        
        self.prelu2 = activation
        
        self.convt3 = nn.ConvTranspose2d(in_channels = self.reduced_channels,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  padding = 0,
                                  bias = False)
        
        self.prelu3 = activation
        
        self.batchnorm = nn.BatchNorm2d(self.reduced_channels)
        self.batchnorm1 = nn.BatchNorm2d(self.reduced_channels)
        self.batchnorm2 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, input_map, indices):
        y = input_map
        
        # Side Branch
        x = self.convt1(input_map)
        x = self.batchnorm(x)
        x = self.prelu1(x)
        
        x = self.convt2(x)
        x = self.batchnorm1(x)
        x = self.prelu2(x)
        
        x = self.convt3(x)
        x = self.batchnorm2(x)
        
        x = self.dropout(x)
        
        # Main Branch
        
        y = self.main_conv(y)
        y = self.unpooling(y, indices, output_size=x.size())
        
        # Concat
        output = x + y
        output = self.prelu3(x)
        
        return output