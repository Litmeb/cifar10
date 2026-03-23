import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import einops
import sys
import ray.train.torch
import tempfile
import os
import ray
sys.path.append('D:\\Git\\cifar10')
from metric import get_metrics

class residual_connection(nn.Module):
    def __init__(self,in_channel,hidden_channel,out_channel,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channel,hidden_channel,kernel_size=1,stride=1,padding=0)
        self.conv2=nn.Conv2d(hidden_channel,hidden_channel,kernel_size,padding=padding)
        self.conv3=nn.Conv2d(hidden_channel,out_channel,kernel_size=1,padding=0)
        self.silu=nn.SiLU()
        self.batchnorm1=nn.BatchNorm2d(in_channel)
        self.batchnorm2=nn.BatchNorm2d(hidden_channel)
        self.batchnorm3=nn.BatchNorm2d(hidden_channel)
    def forward(self,x):
        y=x.clone()
        x=self.batchnorm1(x)
        x=self.conv1(x)
        x=self.silu(x)
        x=self.batchnorm2(x)
        x=self.conv2(x)
        x=self.silu(x)
        x=self.batchnorm3(x)
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
        self.batchnorm1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.resnetblock1=resnetblock(32,16,32,2,5,1,padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.silu1 = nn.SiLU()

        # self.batchnorm2 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.resnetblock2=resnetblock(32,16,32,2,5,1,padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.silu2 = nn.SiLU()

        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.resnetblock3=resnetblock(64,16,64,2,5,1,padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.silu3 = nn.SiLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16, 64)
        self.silu4 = nn.SiLU()
        self.fc2 = nn.Linear(64, 10)
        
    #     # 初始化权重为 Xavier
    #     self._initialize_weights()
    
    # def _initialize_weights(self):
    #     """使用 Xavier 初始化所有卷积层和全连接层的权重"""
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        x=self.batchnorm1(x)
        x=self.conv1(x)
        x=self.resnetblock1(x)
        x=self.pool1(x)
        x=self.silu1(x)
        # x=self.batchnorm2(x)
        x=self.resnetblock2(x)
        x=self.pool2(x)
        x=self.silu2(x)
        x=self.batchnorm3(x)
        x=self.conv3(x)
        x=self.resnetblock3(x)
        x=self.pool3(x)
        x=self.silu3(x)
        x=self.flatten(x)
        x=self.silu4(self.fc1(x))
        x=self.fc2(x)
        return x
def train_func():
    transform=torchvision.transforms.Compose([torchvision.transforms.RandAugment(num_ops=2,magnitude=6),torchvision.transforms.ToTensor()])
    cifar_trainset=torchvision.datasets.CIFAR10('CIFAR10',train=True,transform=transform,download=True)
    cifar_testset=torchvision.datasets.CIFAR10('CIFAR10',train=False,transform=torchvision.transforms.ToTensor(),download=True)
    print(len(cifar_trainset))
    cifar_trainset,cifar_validateset=torch.utils.data.random_split(cifar_trainset,[40000,10000])
    load_trainset=DataLoader(cifar_trainset,batch_size=64,shuffle=True,drop_last=True)
    load_validateset=DataLoader(cifar_validateset,batch_size=64,shuffle=True,drop_last=True)
    load_testset=DataLoader(cifar_testset,batch_size=64,drop_last=False)
    model=resnet()
    model=ray.train.torch.prepare_model(model)
    ray_load_trainset=ray.train.torch.prepare_data_loader(load_trainset)
    ray_load_validateset=ray.train.torch.prepare_data_loader(load_validateset)
    ray_load_testset=ray.train.torch.prepare_data_loader(load_testset)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.002)
    loss=nn.CrossEntropyLoss()
    for epoch in range(2):
        totalloss=0
        for data in ray_load_trainset:
            imgs,label=data
            result=model(imgs)
            result_loss=loss(result,label)
            optimizer.zero_grad()
            result_loss.backward()
            optimizer.step()
            totalloss+=result_loss.item()
        print(f"epoch={epoch},loss={totalloss},",end='')
        with torch.no_grad():
            f1,acc,precision,recall=get_metrics(model,load_validateset)
            print(f"f1={f1.mean()},acc={acc}")
            metrics={'loss':totalloss,'epoch':epoch,'f1':f1.mean(),'acc':acc}
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.save(model.module.state_dict(),os.path.join(temp_dir,'model.pt'))
            ray.train.report(metrics,checkpoint=ray.train.Checkpoint.from_directory(temp_dir))
        if ray.train.get_context().get_world_rank()==0:
            print(f"epoch={epoch},loss={totalloss},f1={f1.mean()},acc={acc}")
if __name__=='__main__':
    ray.init()
    scaling_config=ray.train.ScalingConfig(num_workers=1)
    trainer=ray.train.torch.TorchTrainer(train_func,scaling_config=scaling_config)
    result=trainer.fit()
    with result.checkpoint.as_directory() as checkpoint_dir:
        model_state_dict=torch.load(os.path.join(checkpoint_dir,'model.pt'))
        model=resnet()
        model.load_state_dict(model_state_dict)
        print(model)
