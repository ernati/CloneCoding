import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module) :
    expansion = 1

    # 생성 예시 : block( self.in_planes, planes, stride) 
    # in_planes : 이번 레이어의 입력채널 수 ( 이전 레이어의 출력 채널 수 )
    # planes : 이번 레이어의 출력채널 수
    # stride : 픽셀마다 건너뛰는 정도
    def __init__(self, in_planes, planes, stride=1) :
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
        padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes, kernel_size=3, stride=stride, 
        padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion * planes, 
                               kernel_size=1, bias=False )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()

        # stride가 1이 아닐 경우, 입력값의 크기를 맞춰주기 위해 1x1 컨볼루션을 사용
        if stride != 1 or in_planes != self.expansion * planes :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x) :
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out) + self.shortcut(x))
        out = F.relu(out)
        return out

class ResNet18(nn.Module) :
    def __init__(self, block, num_classes=10) :
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = [2,2,2,2]

        # 3x3 컨볼루션을 사용하여 입력값의 크기를 32x32로 맞춰줌
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # BasicBlock을 2번 반복
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # BasicBlock을 2번 반복
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # BasicBlock을 2번 반복
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # BasicBlock을 2번 반복
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 1x1 컨볼루션을 사용하여 출력값의 크기를 10으로 맞춰줌
        self.linear = nn.Linear(512*block.expansion, num_classes)

    # BasicBlock을 반복하여 레이어를 생성하는 함수
    def _make_layer(self, block, planes, num_blocks, stride) :
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides :
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # 레이어를 순차적으로 연결하여 네트워크 생성
    def forward(self, x) :
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out) # out.shape = [batch_size, 512, 4, 4]
        out = F.avg_pool2d(out, 4 ) # out.shape = [batch_size, 512, 1, 1]
        out = out.view(out.size(0), -1) # out.shape = [batch_size, 512]
        out = self.linear(out) # out.shape = [batch_size, 10]
        return out
