import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import  SummaryWriter
writer=SummaryWriter('logs')
cifar_trainset=torchvision.datasets.CIFAR10('CIFAR10',train=True,transform=torchvision.transforms.ToTensor(),download=True)
cifar_testset=torchvision.datasets.CIFAR10('CIFAR10',train=False,transform=torchvision.transforms.ToTensor(),download=True)
load_trainset=DataLoader(cifar_trainset,batch_size=64,shuffle=True,drop_last=True)
load_teatset=DataLoader(cifar_testset,batch_size=64,drop_last=True)
class cifarmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Sequential(
        nn.Conv2d(3,32,5,padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32,32,5,padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32,64,5,padding=2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64*16,64),
        nn.Linear(64,10))
    def forward(self,x):
        x=self.model(x)
        return x
model=cifarmodel()
if torch.cuda.is_available():
    model=model.cuda()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
loss=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss=loss.cuda()
for epoch in range(30):
    totalloss=0
    for data in load_trainset:
        imgs,label=data
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            label=label.cuda()
        result=model(imgs)
        result_loss=loss(result,label)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        totalloss+=result_loss.item()
    print(f"epoch={epoch},loss={totalloss},",end='')
    writer.add_scalar(tag='loss',scalar_value=totalloss,global_step=epoch)
    with torch.no_grad():
        acc=0
        total=0
        for data in load_teatset:
            total+=64
            imgs,label=data
            if torch.cuda.is_available():
                imgs=imgs.cuda()
                label=label.cuda()
            ans=model(imgs)
            acc+=torch.sum(torch.argmax(ans,axis=1)==label).item()
        print(f"acc={acc/total}")
        writer.add_scalar(tag='acc',scalar_value=acc/total,global_step=epoch)
writer.close()
torch.save(model,'cifar.pth')