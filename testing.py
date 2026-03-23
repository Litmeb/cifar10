import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
try:
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    print("Gloo 初始化成功！")
    dist.destroy_process_group()
except Exception as e:
    print("Gloo 初始化失败:", e)