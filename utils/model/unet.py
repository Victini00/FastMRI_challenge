import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        # Conv. Unet에서는 3x3 필터를 사용한다.
        self.first_block = ConvBlock(in_chans, 2)
        # Down 함수를 통한 다운샘플링
        self.down1 = Down(2, 4)
        # UP 함수를 통한 업샘플링
        self.up1 = Up(4, 2)
        # Conv2d가 중요한 함수다.
        # kernel_size는 커널 가로세로 크기를 의미한다. 여기선 1x1이다.
        # out_chans는 커널의 개수를 의미한다.
        ### 커널의 초기값은 랜덤으로 정해지고, Backpropagation으로 갱신된다. ###
        self.last_block = nn.Conv2d(2, out_chans, kernel_size=1)

    # 정규화.
    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        input, mean, std = self.norm(input)
        # unsqueeze 함수: 지정한 차원 자리에 size가 1인 빈 공간을 추가한다.
        # squeeze 함수는 반대로 size가 1인 공간을 제거한다
        input = input.unsqueeze(1)
        d1 = self.first_block(input)
        m0 = self.down1(d1)
        u1 = self.up1(m0, d1)
        output = self.last_block(u1)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


# U-net에서 오른쪽으로 이동하는 과정.
class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            # 필터를 통한 연산 -> 배치 정규화 -> 활성화 함수를 적용. 총 2번
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


# U-net에서 아래로 내려가는 과정, Max pooling 2x2.
class Down(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            # 2x2 풀링
            nn.MaxPool2d(2),
            ConvBlock(in_chans, out_chans)
        )

    def forward(self, x):
        return self.layers(x)

# U-net에서 위로 올라가는 과정. pooling의 반대 과정을 거친다.
class Up(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        # deconvolution이다.
        # 실제로는, Conv의 역연산이 아니라, Transpose 연산임.
        # Decoder에서도 feature map의 크기를 복원하기 위해 자주 사용된다.
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)