import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import  SummaryWriter
import einops
import sys
sys.path.append('D:\\Git\\cifar10')

from metric import get_metrics
writer=SummaryWriter('logs')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    transform=torchvision.transforms.Compose([torchvision.transforms.RandAugment(num_ops=2,magnitude=6),torchvision.transforms.ToTensor()])
    cifar_trainset=torchvision.datasets.CIFAR10('CIFAR10',train=True,transform=transform,download=True)
    cifar_testset=torchvision.datasets.CIFAR10('CIFAR10',train=False,transform=torchvision.transforms.ToTensor(),download=True)
    print(len(cifar_trainset))
    cifar_trainset,cifar_validateset=torch.utils.data.random_split(cifar_trainset,[40000,10000])
    load_trainset=DataLoader(cifar_trainset,batch_size=64,shuffle=True,drop_last=True)
    load_validateset=DataLoader(cifar_validateset,batch_size=64,shuffle=True,drop_last=True)
    load_testset=DataLoader(cifar_testset,batch_size=64,drop_last=False)
class cifarmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3,32,5,padding=2)
        self.silu1 = nn.SiLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32,5,padding=2)
        self.silu2 = nn.SiLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,5,padding=2)
        self.silu3 = nn.SiLU()
        self.pool3 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*16,64)
        self.silu4 = nn.SiLU()
        self.fc2 = nn.Linear(64,10)
    
    def forward(self,x):
        x = self.batchnorm1(x)
        x = self.conv1(x)
        x = self.silu1(x)
        x = self.pool1(x)
        
        x = self.batchnorm2(x)
        x = self.conv2(x)
        x = self.silu2(x)
        x = self.pool2(x)
        
        x = self.batchnorm3(x)
        x = self.conv3(x)
        x = self.silu3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.silu4(x)
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    model=cifarmodel()
    model=model.to(device)

    f1_record=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],device=device)
    acc_record=torch.tensor([0],device=device)
    precision_record=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],device=device)
    recall_record=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.002)
    loss=nn.CrossEntropyLoss()
    loss=loss.to(device)
    for epoch in range(20):
        totalloss=0
        for data in load_trainset:
            imgs,label=data
            imgs=imgs.to(device)
            label=label.to(device)
            result=model(imgs)
            result_loss=loss(result,label)
            optimizer.zero_grad()
            result_loss.backward()
            optimizer.step()
            totalloss+=result_loss.item()
        print(f"epoch={epoch},loss={totalloss},",end='')
        writer.add_scalar(tag='loss',scalar_value=totalloss,global_step=epoch)
        with torch.no_grad():
            f1,acc,precision,recall=get_metrics(model,load_validateset)
            print(f"f1={f1.mean()},acc={acc}")
            writer.add_scalar(tag='f1',scalar_value=f1.mean(),global_step=epoch)
            writer.add_scalar(tag='acc',scalar_value=acc,global_step=epoch)
            f1_record=torch.cat((f1_record,f1.unsqueeze(0)),dim=0)
            acc_record=torch.cat((acc_record,acc.unsqueeze(0)),dim=0)
            precision_record=torch.cat((precision_record,precision.unsqueeze(0)),dim=0)
            recall_record=torch.cat((recall_record,recall.unsqueeze(0)),dim=0)
        torch.save(model,f'data/cnn_cifar10_{epoch}_{acc.item()}.pth')
    # torch.save((f1_record.detach().clone()),'data/f1_record/cnn.pt')
    # torch.save(acc_record.detach().clone(),'data/acc_record/cnn.pt')
    # torch.save(precision_record.detach().clone(),'data/precision_record/cnn.pt')
    # torch.save(recall_record.detach().clone(),'data/recall_record/cnn.pt')
    writer.close()

