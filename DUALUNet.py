import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m

class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, activation='relu'):
        super(ConvD, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

        self.maxpool2D = nn.MaxPool2d(kernel_size=2)


        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):


        if not self.first:
            x = self.maxpool2D(x)



        #layer 1 conv, bn
        x = self.conv1(x)
        x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = self.conv2(x)
        y = self.bn2(y)
        y = self.activation(y)

        #layer 3 conv, bn
        z = self.conv3(y)
        z = self.bn3(z)
        z = self.activation(z)

        return z

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )


    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class DUALUNet(nn.Module):
    def __init__(self, checkpoint_path=None,ncls=2,args=None) -> None:
        super(DUALUNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        self.args=args
        #self.dropout_rate=0.5
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        #print("self.encoder:",self.encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        #print("blocks:",len(blocks),len(self.encoder.blocks))# 48 48
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)
        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))


      
        self.side1 = nn.Conv2d(64, ncls, kernel_size=1)
        self.side2 = nn.Conv2d(64, ncls, kernel_size=1)
        self.head = nn.Conv2d(64, ncls, kernel_size=1)

       
        n=36
        c=3
        norm = 'bn'
        activation = 'relu'
        self.convd1 = ConvD(c, n, norm, first=True, activation=activation)
        self.convd2 = ConvD(n, 2 * n, norm, activation=activation)
        self.convd3 = ConvD(2 * n, 4 * n, norm, activation=activation)
        self.convd4 = ConvD(4 * n, 8 * n, norm, activation=activation)
        self.convd5 = ConvD(8 * n, 16 * n, norm, activation=activation)
        self.convd6 = ConvD(16 * n, 32 * n, norm, activation=activation)

        #projection
        shape1,shape2,shape3,shape4=144,288,576,1152
        self.proj1=nn.Sequential(
                nn.Conv2d(shape1, shape1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(shape1),
                nn.ReLU(inplace=True),
                nn.Conv2d(shape1, shape1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(shape1),
                nn.ReLU(inplace=True)
        )

        self.proj2=nn.Sequential(
                nn.Conv2d(shape2, shape2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(shape2),
                nn.ReLU(inplace=True),
                nn.Conv2d(shape2, shape2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(shape2),
                nn.ReLU(inplace=True)
        )

        self.proj3=nn.Sequential(
                nn.Conv2d(shape3, shape3, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(shape3),
                nn.ReLU(inplace=True),
                nn.Conv2d(shape3, shape3, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(shape3),
                nn.ReLU(inplace=True)
        )

        self.proj4=nn.Sequential(
                nn.Conv2d(shape4, shape4, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(shape4),
                nn.ReLU(inplace=True),
                nn.Conv2d(shape4, shape4, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(shape4),
                nn.ReLU(inplace=True)
        )

        self.linear1 = nn.Linear(shape1*2, shape1)
        self.linear2 = nn.Linear(shape2*2, shape2)
        self.linear3 = nn.Linear(shape3*2, shape3)
        self.linear4 = nn.Linear(shape4*2, shape4)




    def forward(self, x,dropoutflag=0):
        x10, x20, x30, x40 = self.encoder(x)
        x00 = self.convd1(x)
        x000 = self.convd2(x00)

        if self.args.model_type == 'few_two_concat_linear':
         
            bs = x10.shape[0]
            shape1, shape2, shape3, shape4 = 144, 288, 576, 1152

            x11 = self.convd3(x000)
            x1 = torch.cat((x10, x11), dim=1)
          
            x1 = x1.permute(0, 2, 3, 1).reshape(-1, shape1 * 2)
          
            x1 = self.linear1(x1)
         
            x1 = x1.view(bs, 48, 48, shape1).permute(0, 3, 1, 2)
         
            x1 = self.proj1(x1)
          
            if dropoutflag==1:
           
               x1 = nn.Dropout2d(p=self.args.dropout_rate)(x1)


            x22 = self.convd4(x1)
            x2 = torch.cat((x20, x22), dim=1)
            x2 = x2.permute(0, 2, 3, 1).reshape(-1, shape2 * 2)
            x2 = self.linear2(x2)
            x2 = x2.view(bs, 24, 24, shape2).permute(0, 3, 1, 2)
            x2 = self.proj2(x2)
            if dropoutflag == 1:
            
                x2 = nn.Dropout2d(p=self.args.dropout_rate)(x2)

            x33 = self.convd5(x2)
            x3 = torch.cat((x30, x33), dim=1)
            x3 = x3.permute(0, 2, 3, 1).reshape(-1, shape3 * 2)
            x3 = self.linear3(x3)
            x3 = x3.view(bs, 12, 12, shape3).permute(0, 3, 1, 2)
            x3 = self.proj3(x3)
            if dropoutflag == 1:
              
                x3 = nn.Dropout2d(p=self.args.dropout_rate)(x3)


            x44 = self.convd6(x3)
            x4 = torch.cat((x40 , x44), dim=1)
            x4 = x4.permute(0, 2, 3, 1).reshape(-1, shape4 * 2)
            x4 = self.linear4(x4)
            x4 = x4.view(bs, 6, 6, shape4).permute(0, 3, 1, 2)
            x4 = self.proj4(x4)
            if dropoutflag == 1:
             
                x4 = nn.Dropout2d(p=self.args.dropout_rate)(x4)

            x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)


        
        else:
            x1, x2, x3, x4 = self.rfb1(x10), self.rfb2(x20), self.rfb3(x30), self.rfb4(x40)



        y1 = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(y1), scale_factor=16, mode='bilinear')
        y2 = self.up2(y1, x2)
        out2 = F.interpolate(self.side2(y2), scale_factor=8, mode='bilinear')
        y3 = self.up3(y2, x1)
        out = F.interpolate(self.head(y3), scale_factor=4, mode='bilinear')

        return out, out1, out2
