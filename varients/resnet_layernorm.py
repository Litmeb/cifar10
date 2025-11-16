import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import  SummaryWriter
import einops
import sys
sys.path.append('D:\\Git\\cifar10')

from metric import get_metrics
writer=SummaryWriter('resnet_logs_layernorm')

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

class residual_connection(nn.Module):
    def __init__(self,in_channel,hidden_channel,out_channel,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channel,hidden_channel,kernel_size,stride,padding=padding)
        self.conv2=nn.Conv2d(hidden_channel,hidden_channel,kernel_size,padding=padding)
        self.conv3=nn.Conv2d(hidden_channel,out_channel,kernel_size,padding=padding)
        self.silu=nn.SiLU()
        self.layernorm1=nn.LayerNorm((in_channel,))
        self.layernorm2=nn.LayerNorm((hidden_channel,))
        self.layernorm3=nn.LayerNorm((hidden_channel,))
    def forward(self,x):
        y=x.clone()
        # LayerNorm expects (N, C, H, W), normalized_shape should be (C,)
        # For 2D input, we need to transpose: (N, C, H, W) -> (N, H, W, C) for LayerNorm
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm1(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x=self.conv1(x)
        x=self.silu(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x=self.conv2(x)
        x=self.silu(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm3(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x=self.conv3(x)
        x=self.silu(x)
        x=x+y
        return x
class resnetblock(nn.Module):
    def __init__(self,in_channel,hidden_channel,out_channel,num_residual_connection=2,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.residual_connections=nn.ModuleList([residual_connection(in_channel,hidden_channel,out_channel,kernel_size,stride,padding) for _ in range(num_residual_connection)])
    def forward(self,x):
        for residual_connection in self.residual_connections:
            x=residual_connection(x)
        return x
class resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm1 = nn.LayerNorm((3,))
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.resnetblock1=resnetblock(32,16,32,2,5,1,padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.silu1 = nn.SiLU()

        # self.layernorm2 = nn.LayerNorm((32,))
        # self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.resnetblock2=resnetblock(32,16,32,2,5,1,padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.silu2 = nn.SiLU()

        self.layernorm3 = nn.LayerNorm((32,))
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.resnetblock3=resnetblock(64,16,64,2,5,1,padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.silu3 = nn.SiLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16, 64)
        self.silu4 = nn.SiLU()
        self.fc2 = nn.Linear(64, 10)
    def forward(self,x):
        # LayerNorm for 2D input: transpose to (N, H, W, C), apply LayerNorm, then transpose back
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm1(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x=self.conv1(x)
        x=self.resnetblock1(x)
        x=self.pool1(x)
        x=self.silu1(x)
        # x=self.layernorm2(x)
        x=self.resnetblock2(x)
        x=self.pool2(x)
        x=self.silu2(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm3(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x=self.conv3(x)
        x=self.resnetblock3(x)
        x=self.pool3(x)
        x=self.silu3(x)
        x=self.flatten(x)
        x=self.silu4(self.fc1(x))
        x=self.fc2(x)
        return x
if __name__ == '__main__':
    model=resnet()
    model=model.to(device)

    f1_record=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],device=device)
    acc_record=torch.tensor([0],device=device)
    precision_record=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],device=device)
    recall_record=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.005)
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
        # torch.save(model,f'data/resnet_cifar10_{epoch}_{acc.item()}.pth')
    torch.save(model,'resnet_layernorm.pth')
    f1,acc,precision,recall=get_metrics(model,load_testset)
    print(f1.mean(),acc)
    # torch.save((f1_record.detach().clone()),'data/f1_record/resnet.pt')
    # torch.save(acc_record.detach().clone(),'data/acc_record/resnet.pt')
    # torch.save(precision_record.detach().clone(),'data/precision_record/resnet.pt')
    # torch.save(recall_record.detach().clone(),'data/recall_record/resnet.pt')
    writer.close()

