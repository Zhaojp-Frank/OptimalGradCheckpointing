#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from solver import optimal_grad_checkpointing
from tqdm import tqdm


# # Define Pytorch Model and Specify Input size and cuda device
# Here we use resnet_cifar10 model as an example

# In[2]:


from torchvision.models.resnet import BasicBlock, conv1x1

class ResNetCifar10(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCifar10, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=1,
                                base_width=64, dilation=1,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def resnet20():
    return ResNetCifar10(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNetCifar10(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNetCifar10(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNetCifar10(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNetCifar10(BasicBlock, [18, 18, 18])


# # Create Pytorch object, specify Input size and cuda device

# In[3]:


device = 'cuda:0'
input_size = (64, 3, 32, 32)
model = resnet20().to(device)


# # Run Optimal Grad Checkpointing
# Create a dummy input to automatically build the computation graph and run optimal grad checkpointing. The returned run_segment is a pytorch nn.Module whose forward function will execute gradient checkpointing training using the optimal grad checkpoints

# In[4]:


inp = torch.randn(*input_size).to(device)
run_segment = optimal_grad_checkpointing(model, inp)

print(type(run_segment))

# # Create CIFAR-10 dataset and data loader

# In[5]:


transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# dir of cifar-10-python.tar.gz
trainset = torchvision.datasets.CIFAR10(root='/disk2/zhaojp', train=True,
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=input_size[0],
                                              shuffle=True, num_workers=0)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=input_size[0], shuffle=False, num_workers=0)


# # Set up loss and optimizer, train and evaluate for two epochs
# The only difference with regular training is that we use run_segment to perform checkpointing training in training phase.

# In[6]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# train for 2 epoch and eval for 2 epoch
for epoch in range(1):
    # use model to switch between train and evaluation
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        optimizer.zero_grad()

        # use run_segment to do checkpointing forward and backward for training
        outputs = run_segment(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] Train loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    '''
    model.eval()
    eval_running_loss = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # use model to do forward for evaluation
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        eval_running_loss += loss.item()
    print('[%d] Eval loss: %.3f' % (epoch + 1, eval_running_loss / len(testloader)))

    # save model weights
    torch.save(model.state_dict(), './checkpoint.pth')
    '''


# In[ ]:




