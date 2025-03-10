from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image
from torchvision import transforms 
import torchvision

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
    train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
    test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)
    image,label=train_set[0]
    image.show()
    print(train_set.classes[label])
    
#     workapce_folder_path="/home/g/workspace/TinyAD"
#     input_folder_path=os.path.join(workapce_folder_path,"dataset","hymenoptera_data","train")
#     data_loader=HymenopteraDataset(input_folder_path,"ants")
#     image_tensor,label=data_loader[0]
#     #image_tensor可视化
#     to_pil_image = transforms.ToPILImage()
#     pil_image = to_pil_image(image_tensor)
#     pil_image.show()
#     print(label)
