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
class ViT(nn.Module):
    def __init__(self,patch1=4,patch2=4,channel=3):
        super().__init__()
        self.patch1=patch1
        self.patch2=patch2
        self.tokenize=nn.Linear(patch1*patch2*channel,64)
        self.embedding=nn.parameter.Parameter(torch.randn(65,64,device=device),requires_grad=True)
        self.classembedding=nn.parameter.Parameter(torch.randn(64,device=device),requires_grad=True)
        self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,dim_feedforward=64,batch_first=True),num_layers=4)
        self.fc=nn.Linear(64,10)
    def forward(self,x):
        x=einops.rearrange(x,'batch channel (h patch1) (w patch2) -> batch channel h w (patch1 patch2)',patch1=self.patch1,patch2=self.patch2)
        x=einops.rearrange(x,'batch channel h w patch -> batch (h w) (channel patch)')
        x=self.tokenize(x)
        x=torch.cat((self.classembedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0],-1,-1),x),dim=1)
        x=x+self.embedding.unsqueeze(0).expand(x.shape[0],-1,-1)
        x=self.transformer(x)
        x=self.fc(x[:,0])
        return x
if __name__ == '__main__':
    model=ViT()
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
        torch.save(model,f'data/ViT_cifar10_{epoch}_{acc.item()}.pth')
    # torch.save((f1_record.detach().clone()),'data/f1_record/ViT.pt')
    # torch.save(acc_record.detach().clone(),'data/acc_record/ViT.pt')
    # torch.save(precision_record.detach().clone(),'data/precision_record/ViT.pt')
    # torch.save(recall_record.detach().clone(),'data/recall_record/ViT.pt')
    writer.close()

