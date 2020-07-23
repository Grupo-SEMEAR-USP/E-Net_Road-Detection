import torch
import torch.nn as nn
from InitialBlock import InitialBlock
from RegularBottleneck import RegularBottleneck
from DownsamplingBottleneck import DownsamplingBottleneck
from AsymmetricBottleneck import AsymmetricBottleneck
from UpsamplingBottleneck import UpsamplingBottleneck

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import models
from torchsummary import summary
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class ENet(nn.Module):
    def __init__(self, Out_Channels):

        super().__init__()

        self.Out_Channels = Out_Channels
        #Bloco inicial
        self.init_block = InitialBlock()

        #Primeira camada de bottlenecks
        self.bottleneck10 = DownsamplingBottleneck(in_channels=16,
                                                    out_channels=64,
                                                    p_dropout=0.01)

        self.bottleneck11 = RegularBottleneck(in_channels=64,
                                                out_channels=64,
                                                p_dropout=0.01)

        self.bottleneck12 = RegularBottleneck(in_channels=64,
                                                out_channels=64,
                                                p_dropout=0.01)

        self.bottleneck13 = RegularBottleneck(in_channels=64,
                                                out_channels=64,
                                                p_dropout=0.01)

        self.bottleneck14 = RegularBottleneck(in_channels=64,
                                                out_channels=64,
                                                p_dropout=0.01)
        
        #Segunda camada de bottlenecks
        self.bottleneck20 = DownsamplingBottleneck(in_channels=64,
                                                    out_channels=128)

        self.bottleneck21 = RegularBottleneck(in_channels=128,
                                                out_channels=128)
        
        self.bottleneck22 = RegularBottleneck(in_channels=128,
                                                out_channels=128,
                                                dilation=2)
        
        self.bottleneck23 = AsymmetricBottleneck(128,128)

        self.bottleneck24 = RegularBottleneck(in_channels=128,
                                                out_channels=128,
                                                dilation=4)

        self.bottleneck25 = RegularBottleneck(in_channels=128,
                                                out_channels=128)

        self.bottleneck26 = RegularBottleneck(in_channels=128,
                                                out_channels=128,
                                                dilation=8)
        
        self.bottleneck27 = AsymmetricBottleneck(128,128)

        self.bottleneck28 = RegularBottleneck(in_channels=128,
                                                out_channels=128,
                                                dilation=16)
        
        #Terceira camada de bottlenecks
        self.bottleneck31 = RegularBottleneck(in_channels=128,
                                                out_channels=128)
        
        self.bottleneck32 = RegularBottleneck(in_channels=128,
                                                out_channels=128,
                                                dilation=2)
        
        self.bottleneck33 = AsymmetricBottleneck(128,128)

        self.bottleneck34 = RegularBottleneck(in_channels=128,
                                                out_channels=128,
                                                dilation=4)

        self.bottleneck35 = RegularBottleneck(in_channels=128,
                                                out_channels=128)

        self.bottleneck36 = RegularBottleneck(in_channels=128,
                                                out_channels=128,
                                                dilation=8)
        
        self.bottleneck37 = AsymmetricBottleneck(128,128)

        self.bottleneck38 = RegularBottleneck(in_channels=128,
                                                out_channels=128,
                                                dilation=16)
        
        #Quarta camada de bottlenecks
        self.bottleneck40 = UpsamplingBottleneck(in_channels=128,
                                                    out_channels=64,
                                                    relu=True)
        
        self.bottleneck41 = RegularBottleneck(in_channels=64,
                                                out_channels=64,
                                                relu=True)
        
        self.bottleneck42 = RegularBottleneck(in_channels=64,
                                                out_channels=64,
                                                relu=True)
        
        #Quinta camada de bottlenecks
        self.bottleneck50 = UpsamplingBottleneck(in_channels=64,
                                                    out_channels=16,
                                                    relu=True)
        
        self.bottleneck51 = RegularBottleneck(in_channels=16,
                                                out_channels=16,
                                                relu=True)
        
        self.fullconv = nn.ConvTranspose2d(in_channels=16,
                                            out_channels=self.Out_Channels,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=False)
        #Função de ativação que irá mapear pixels entre 0 a 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_image):
        #Bloco inicial
        x = self.init_block(input_image)
        #Primeira camada
        x, i1 = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        x = self.bottleneck14(x)
        #Segunda camada
        x, i2 = self.bottleneck20(x)
        x = self.bottleneck21(x)
        x = self.bottleneck22(x)
        x = self.bottleneck23(x)
        x = self.bottleneck24(x)
        x = self.bottleneck25(x)
        x = self.bottleneck26(x)
        x = self.bottleneck27(x)
        x = self.bottleneck28(x)
        #Terceira camada
        x = self.bottleneck31(x)
        x = self.bottleneck32(x)
        x = self.bottleneck33(x)
        x = self.bottleneck34(x)
        x = self.bottleneck35(x)
        x = self.bottleneck36(x)
        x = self.bottleneck37(x)
        x = self.bottleneck38(x)
        #Quarta camada
        x = self.bottleneck40(x, i2)
        x = self.bottleneck41(x)
        x = self.bottleneck42(x)
        #Quinta camada
        x = self.bottleneck50(x, i1)
        x = self.bottleneck51(x)
        #Camada transposta final
        output_image = self.fullconv(x)

        output_image = self.sigmoid(output_image)

        return output_image

'''def _initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean=0, std=0.02)

def print_models():
    print("E-Net model: \n\n")
    #Queremos um canal na saída
    E_net = ENet(1)
    E_net.apply(_initialize_weights)
    E_net.cuda()
    summary(E_net, (3, 512, 512))

print_models()'''




