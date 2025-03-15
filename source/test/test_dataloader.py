import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

#测试集
test_data=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)


test_loader=torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=False)
writer=SummaryWriter("logs")

step=0
for epoch in range(2):
    for data in test_loader:
        imgs,targets=data
        writer.add_images(f"test_data_{epoch}",imgs,step)
        step=step+1
    writer.close()