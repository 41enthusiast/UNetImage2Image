from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .model_utils import *

class DecoderBlock_UNetpcb(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=1):        
        super().__init__()
        if upsample:
            # self.upconv = nn.ConvTranspose2d(in_channels*2, in_channels*2, kernel_size=2, stride=2)
            self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upconv = nn.Identity()
        self.layers = DoubleConv(in_channels * 2, out_channels, out_channels)
        
    def forward(self, x, skip_connection, zero_conv=None):
 
        target_height = x.size(2)
        target_width = x.size(3)
        skip_interp = F.interpolate(
            skip_connection, size=(target_height, target_width), mode='bilinear', align_corners=False)
        zero_conved = zero_conv(skip_interp) if zero_conv is not None else skip_interp
        
        concatenated = torch.cat([zero_conved,  x], dim=1)   

        concatenated = self.upconv(concatenated)
            
        output = self.layers(concatenated)
        return output

class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),  
    )
    def forward(self, inputs):
        return self.layers(inputs)

class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid() 
    )
    def forward(self, inputs):
        return self.layers(inputs)

class UNet_pcb(nn.Module):
    def __init__(self, num_channels = 1, pretrained=True,
                input_features=3, layer1_features=32, layer2_features=16,
                layer3_features=24, layer4_features=40, layer5_features=80):
        """
        pretrained classifier backbone based on EfficientNet-B0
        """
        super(UNet_pcb, self).__init__()
        self.effnet = models.efficientnet_b0(pretrained=pretrained)

        self.num_classes = num_channels #get mask output (gray scale)

#         # Layer feature sizes
        self.input_features = input_features
        self.layer1_features = layer1_features
        self.layer2_features = layer2_features
        self.layer3_features = layer3_features
        self.layer4_features = layer4_features
        self.layer5_features = layer5_features
        
#         Encoder layers
        self.encoder1 = nn.Sequential(*list(self.effnet.features.children())[0])  #out 32,112*112
        self.encoder2 = nn.Sequential(*list(self.effnet.features.children())[1])  #out 16,112*112
        self.encoder3 = nn.Sequential(*list(self.effnet.features.children())[2])  #out 24,56*56
        self.encoder4 = nn.Sequential(*list(self.effnet.features.children())[3])  #out 40,28*28
        self.encoder5 = nn.Sequential(*list(self.effnet.features.children())[4])  #out 40,28*28

        # Zero Convolutions for skip connections for softer integration of efficientnet features
        self.zero_conv1 = nn.Conv2d(self.layer1_features, self.layer1_features, kernel_size=1)
        self.zero_conv2 = nn.Conv2d(self.layer2_features, self.layer2_features, kernel_size=1)
        self.zero_conv3 = nn.Conv2d(self.layer3_features, self.layer3_features, kernel_size=1)
        self.zero_conv4 = nn.Conv2d(self.layer4_features, self.layer4_features, kernel_size=1)
        self.zero_conv5 = nn.Conv2d(self.layer5_features, self.layer5_features, kernel_size=1)
        nn.init.normal_(self.zero_conv1.weight, mean=0.0, std=1e-3), nn.init.zeros_(self.zero_conv1.bias)
        nn.init.normal_(self.zero_conv2.weight, mean=0.0, std=1e-3), nn.init.zeros_(self.zero_conv2.bias)
        nn.init.normal_(self.zero_conv3.weight, mean=0.0, std=1e-3), nn.init.zeros_(self.zero_conv3.bias)
        nn.init.normal_(self.zero_conv4.weight, mean=0.0, std=1e-3), nn.init.zeros_(self.zero_conv4.bias)
        nn.init.normal_(self.zero_conv5.weight, mean=0.0, std=1e-3), nn.init.zeros_(self.zero_conv5.bias)

        
        del self.effnet
        
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False

        # Bottleneck Layer
        self.bottleneck = DoubleConv(self.layer5_features, self.layer5_features, self.layer5_features)   
        
        # Decoder layers
        self.decoder1 = DecoderBlock_UNetpcb(self.layer5_features, self.layer4_features)
        self.decoder2 = DecoderBlock_UNetpcb(self.layer4_features, self.layer3_features)
        self.decoder3 = DecoderBlock_UNetpcb(self.layer3_features, self.layer2_features)
        self.decoder4 = DecoderBlock_UNetpcb(self.layer2_features, self.layer1_features, upsample=0)
        self.decoder5 = DecoderBlock_UNetpcb(self.layer1_features, self.layer1_features)        
        # Final layer
        self.final_conv = FinalLayer(self.layer1_features, self.num_classes)
        
        
    def forward(self, x):
        # Encoder (contracting path)
        output1 = self.encoder1(x)
        output2 = self.encoder2(output1)
        output3 = self.encoder3(output2)
        output4 = self.encoder4(output3)
        output5 = self.encoder5(output4)
        
        # Bottleneck Layer
        bn = self.bottleneck(output5)
        up1 = self.decoder1(bn,  output5, zero_conv=self.zero_conv5)
        up2 = self.decoder2(up1, output4, zero_conv=self.zero_conv4)
        up3 = self.decoder3(up2, output3, zero_conv=self.zero_conv3)
        up4 = self.decoder4(up3, output2, zero_conv=self.zero_conv2)
        up5 = self.decoder5(up4, output1, zero_conv=self.zero_conv1) 
        
        # Final convolution to produce segmentation mask
        res = self.final_conv(up5)

        return res


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_gpu()
    model = UNet_pcb().to(device)
    print(summary(model, (3, 256, 256)))