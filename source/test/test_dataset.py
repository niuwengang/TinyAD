from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms 
import torchvision
from torch.utils.tensorboard import SummaryWriter

class HymenopteraDataset(Dataset):
    def __init__(self,input_folder_path,label):
        super().__init__()
        self.input_folder_path=input_folder_path
        self.label=label
        self.image_path_list=os.listdir(os.path.join(self.input_folder_path,self.label))


    def __getitem__(self, index):
        image_name=self.image_path_list[index]
        image_file_path=os.path.join(self.input_folder_path,self.label,image_name)
        #读取图片
        image=Image.open(image_file_path)
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        return image_tensor,self.label


if __name__=="__main__":
    dataset_transform=torchvision.transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
    test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)
    print(train_set[0])

    writer=SummaryWriter("p10")
    for i in range(10):
        img,target=train_set[i]
        writer.add_image("train",img,i)
    writer.close()
    #tensorboard --logdir=p10
    
#     workapce_folder_path="/home/g/workspace/TinyAD"
#     input_folder_path=os.path.join(workapce_folder_path,"dataset","hymenoptera_data","train")
#     data_loader=HymenopteraDataset(input_folder_path,"ants")
#     image_tensor,label=data_loader[0]
#     #image_tensor可视化
#     to_pil_image = transforms.ToPILImage()
#     pil_image = to_pil_image(image_tensor)
#     pil_image.show()
#     print(label)
