import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AsymmetricBottleneck(nn.Module):

    def __init__ (self,in_channels, out_channels, projection_ratio_conv=4, relu = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()
        
        self.reduced_channels = int(in_channels//projection_ratio_conv)

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                                out_channels=self.reduced_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)

        self.activation_function1 = activation


        self.asymmetric_conv1 = nn.Conv2d(in_channels=self.reduced_channels,
                                    out_channels=self.reduced_channels,
                                    kernel_size=(1,5),
                                    stride=1,
                                    padding=(0,2),
                                    bias=False)

        self.asymmetric_conv2 = nn.Conv2d(in_channels=self.reduced_channels,
                                    out_channels=self.reduced_channels,
                                    kernel_size=(5,1),
                                    stride=1,
                                    padding=(2,0),
                                    bias=False)

        self.activation_function2 = activation

        self.conv2 = nn.Conv2d(in_channels=self.reduced_channels, 
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)

        self.batch_norm = nn.BatchNorm2d(self.reduced_channels)
        
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.activation_function3 = activation

        self.dropout = nn.Dropout2d(p=0.1)        
    
    def forward(self, input_map):
        dimension = input_map.size()[0]

        #Ramo lateral
        x = self.conv1(input_map)
        x = self.batch_norm(x)
        x = self.activation_function1(x)
        x = self.asymmetric_conv1(x)
        x = self.asymmetric_conv2(x)
        x = self.batch_norm(x)
        x = self.activation_function2(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)

        #Ramo principal
        y = input_map

        if self.in_channels != self.out_channels:
            out_channels_needed = self.out_channels - self.in_channels
            tensor_extra = torch.zeros((dimension,out_channels_needed,x.shape[2],x.shape[3]))
            tensor_extra = tensor_extra.to(device)
            y = torch.cat((y,tensor_extra),dim=1)

        #Soma dos elementos
        output = x + y
        output = self.activation_function3(output)

        return output