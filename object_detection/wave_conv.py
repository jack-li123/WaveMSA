from torch import nn
import torch
from DWT_IDWT1 import DWT_IDWT_layer
from torch_wavelets import DWT_2D,IDWT_2D

class ConvBNReLU1x1_br(nn.Module):
    """
       实现一个包含1x1卷积、批量归一化和ReLU激活函数的模块，特别地，该模块在卷积之前和之后应用了一维离散小波变换(DWT)。
        要去通道数应为偶数
       参数:
       - in_channel: 输入通道数
       - out_channel: 输出通道数
       - stride: 卷积步长
       - kernel_size: 卷积核大小，默认为1
       - padding: 卷积补齐大小，默认为0
       - groups: 卷积组数，默认为1
       - bias: 是否使用偏置，默认为False
       """
    def __init__(self, in_channel, out_channel, stride, padding=0, groups=1, bias=False):
        super(ConvBNReLU1x1_br, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channel // 2, out_channel // 2,kernel_size=1,stride=stride, padding=padding,
                                  groups=groups, bias=bias)
        self.dwt_1D = DWT_IDWT_layer.DWT_1D('haar')
        self.idwt_1D = DWT_IDWT_layer.IDWT_1D('haar')
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU6(inplace=True)

    def forward(self, x):
        def process_component(component,B,H,W,C):
            # 调整形状以便应用1x1卷积
            component = component.view(B, H, W, C // 2).permute(0, 3, 1, 2)
            component = self.conv_1x1(component)
            _, Ctemp, _, _ = component.shape
            component = component.permute(0, 2, 3, 1).view(B, H * W, Ctemp)
            return component

        B, C, W, H = x.shape
        # 确保C是偶数，以便正确分割为低频和高频组件
        assert C % 2 == 0, "通道数C应当为偶数"
        x = x.permute(0, 2, 3, 1).view(B, W * H, C)
        Low, High = self.dwt_1D(x)

        Low = process_component(Low,B,H,W,C)
        High = process_component(High,B,H,W,C)

        x = self.idwt_1D(Low, High)
        _, _, Ctemp01 = x.shape
        x = x.view(B, H, W, Ctemp01).permute(0, 3, 1, 2)
        x=self.bn(x)
        x = self.relu(x)
        return x


class ConvBNReLU1x1(nn.Module):
    """
       实现一个包含1x1卷积、批量归一化和ReLU激活函数的模块，特别地，该模块在卷积之前和之后应用了一维离散小波变换(DWT)。
        要去通道数应为偶数
       参数:
       - in_channel: 输入通道数
       - out_channel: 输出通道数
       - stride: 卷积步长
       - kernel_size: 卷积核大小，默认为1
       - padding: 卷积补齐大小，默认为0
       - groups: 卷积组数，默认为1
       - bias: 是否使用偏置，默认为False
       """

    def __init__(self, in_channel, out_channel, stride, padding=0, groups=1, bias=False):
        super(ConvBNReLU1x1, self).__init__()
        self.conv_1x1_0 = nn.Conv2d(in_channel // 2, out_channel // 2, kernel_size=1, stride=stride, padding=padding,
                                  groups=groups, bias=bias)
        self.conv_1x1_1 = nn.Conv2d(in_channel // 2, out_channel // 2, kernel_size=1, stride=stride, padding=padding,
                                  groups=groups, bias=bias)
        self.dwt_1D = DWT_IDWT_layer.DWT_1D('haar')
        self.idwt_1D = DWT_IDWT_layer.IDWT_1D('haar')

    def forward(self, x):

        B, C, W, H = x.shape
        # 确保C是偶数，以便正确分割为低频和高频组件
        assert C % 2 == 0, "通道数C应当为偶数"
        x = x.permute(0, 2, 3, 1).view(B, W * H, C)
        #B,W,H,C - >B,W*H,C
        Low, High = self.dwt_1D(x)



        Low=Low.view(B, H, W, C // 2).permute(0, 3, 1, 2)
        Low=self.conv_1x1_0(Low)
        _, Ctemp, _, _ = Low.shape
        Low=Low.permute(0, 2, 3, 1).view(B, H * W, Ctemp)


        High = High.view(B, H, W, C // 2).permute(0, 3, 1, 2)
        High = self.conv_1x1_1(High)
        _, Ctemp, _, _ = High.shape
        High = High.permute(0, 2, 3, 1).view(B, H * W, Ctemp)

        x = self.idwt_1D(Low, High)
        #B, H*W, C//4
        _, _, Ctemp01 = x.shape
        x = x.view(B, H, W, Ctemp01).permute(0, 3, 1, 2)
        #B, H*W, C//4 ->
        return x


class WaveConv(nn.Module):
    """
       实现一个包含1x1卷积、批量归一化和ReLU激活函数的模块，特别地，该模块在卷积之前和之后应用了一维离散小波变换(DWT)。
        要去通道数应为偶数
       参数:
       - in_channel: 输入通道数
       - out_channel: 输出通道数
       - stride: 卷积步长
       - kernel_size: 卷积核大小，默认为1
       - padding: 卷积补齐大小，默认为0
       - groups: 卷积组数，默认为1
       - bias: 是否使用偏置，默认为False
       """

    def __init__(self):
        super(WaveConv, self).__init__()
        self.dwt_1D = DWT_IDWT_layer.DWT_1D('haar')
        self.idwt_1D = DWT_IDWT_layer.IDWT_1D('haar')

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保C是偶数，以便正确分割为低频和高频组件
        assert C % 2 == 0, "通道数C应当为偶数"
        x = x.permute(0, 2, 3, 1).view(B, W * H, C)
        Low, High = self.dwt_1D(x)
        LL,LH=self.dwt_1D(Low)
        HL,HH=self.dwt_1D(High)
        x=LL
        # x = self.idwt_1D(Low, High)
        _, _, Ctemp01 = x.shape
        x = x.view(B, H, W, Ctemp01).permute(0, 3, 1, 2)
        return x


class ConvBNReLU3x3(nn.Sequential):
    """
      一个包含3x3卷积、批归一化和ReLU激活函数的模块，特别地，它还整合了二维离散小波变换(DWT)。
        要求宽度高度为偶数且输入通道等于输出通道
      参数:
      - in_channel (int): 输入通道数。
      - out_channel (int): 输出通道数。
      - stride (int): 卷积步长。
      - padding (int, optional): 卷积补齐大小，默认为1。
      - bias (bool, optional): 是否在卷积中使用偏置，默认为False。
      """
    def __init__(self, in_channel, out_channel, stride, padding=1, bias=False):
        super(ConvBNReLU3x3, self).__init__()
        self.conv_3x3=nn.Conv2d(in_channel, out_channel, stride=stride,kernel_size=3, groups=in_channel,bias=bias,padding=padding)
        self.dwt_2D = DWT_IDWT_layer.DWT_2D('haar')
        self.idwt_2D = DWT_IDWT_layer.IDWT_2D('haar')
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU6(inplace=True)

    def forward(self, x):
        _, _, _, H = x.shape
        assert H % 2 == 0, "宽高应当为偶数"
        LL, LH, HL, HH = self.dwt_2D(x)
        LL = self.conv_3x3(LL)
        LH = self.conv_3x3(LH)
        HL = self.conv_3x3(HL)
        HH = self.conv_3x3(HH)
        x = self.idwt_2D(LL, LH, HL, HH)
        x=self.bn(x)
        x = self.relu(x)

        return x

class ConvBNReLU3x3_conacat(nn.Sequential):
    """
      一个包含3x3卷积、批归一化和ReLU激活函数的模块，特别地，它还整合了二维离散小波变换(DWT)。
        要求宽度高度为偶数且输入通道等于输出通道
      参数:
      - in_channel (int): 输入通道数。
      - out_channel (int): 输出通道数。
      - stride (int): 卷积步长。
      - padding (int, optional): 卷积补齐大小，默认为1。
      - bias (bool, optional): 是否在卷积中使用偏置，默认为False。
      """
    def __init__(self, in_channel, out_channel, stride, padding=1, bias=False):
        super(ConvBNReLU3x3_conacat, self).__init__()
        self.conv_3x3=nn.Conv2d(in_channel, out_channel, stride=stride,kernel_size=3, groups=in_channel,bias=bias,padding=padding)
        self.dwt_2D = DWT_IDWT_layer.DWT_2D('haar')
        self.idwt_2D = DWT_IDWT_layer.IDWT_2D('haar')
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        _, _, _, H = x.shape
        assert H % 2 == 0, "宽高应当为偶数"
        LL, LH, HL, HH = self.dwt_2D(x)
        x = torch.cat([LL, LH, HL, HH], dim=1)
        x = self.conv_3x3(x)
        x = self.idwt_2D(x)
        x=self.bn(x)
        x = self.relu(x)
        return x



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )



if __name__ == '__main__':
    dim=64
    model = WaveConv()
    inputs = torch.randn(1, 64, 56, 56)
    outputs = model(inputs)
    print(outputs.shape)