from torch.utils.data import Dataset
import cv2
import os 
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import plotly.express as px

def visualize_tensor_plotly(tensor, title="Tensor Visualization", color_scale='viridis'):
    """
    使用 Plotly 可视化 PyTorch 张量图像。

    参数:
        tensor (torch.Tensor): 输入的 PyTorch 张量，形状为 (C, H, W)。
        title (str): 图像标题，默认为 "Tensor Visualization"。
        color_scale (str): Plotly 的颜色映射，默认为 'viridis'。

    返回:
        None: 直接显示图像。
    """
    # 检查输入是否为张量
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch Tensor.")
    
    # 确保张量的形状为 (C, H, W)
    if tensor.ndim != 3:
        raise ValueError("Input tensor must have shape (C, H, W).")
    
    # 转换张量形状为 (H, W, C)
    tensor_permuted = tensor.permute(1, 2, 0)
    
    # 转换为 NumPy 数组
    image_np = tensor_permuted.numpy()
    
    # 使用 Plotly 进行可视化
    fig = px.imshow(image_np, color_continuous_scale=color_scale)
    fig.update_layout(title=title)
    fig.show()

class MyData(Dataset):
    def __init__(self,root_dir,label):
        self.root_dir=root_dir
        self.label=label
        self.object_folder_path=os.path.join(self.root_dir,self.label+"_image")
        self.label_folder_path=os.path.join(self.root_dir,self.label+"_label")
        self.object_list=os.listdir(self.object_folder_path)
    def __getitem__(self, idx):
        object_name=self.object_list[idx] 
        object_path=os.path.join(self.object_folder_path,object_name)
        image=cv2.imread(object_path)
        label=self.label
        return image,label
    def __len__(self):
        return len(self.object_list)

#数据读取 
def Usage1():
    root_dir="data/hymenoptera_data"
    ants_label_dir="ants"
    bees_label_dir="bees"
    ants_dataset=MyData(root_dir,ants_label_dir)
    bees_dataset=MyData(root_dir,bees_label_dir)
    train_dataset=ants_dataset+bees_dataset
    print(len(train_dataset))
    image,label =train_dataset[120]
    cv2.imshow("image",image)  
    cv2.waitKey()

#可视化看板
def Usage2():
    image1=cv2.imread("data/hymenoptera_data/ants_image/0013035.jpg")
    image2=cv2.imread("data/hymenoptera_data/ants_image/5650366_e22b7e1065.jpg")
    writer=SummaryWriter("logs")
    writer.add_image("test",image1,1,dataformats='HWC')
    writer.add_image("test",image2,2,dataformats='HWC')
    for i in range(100):
            writer.add_scalar("y=x",i,i)
            writer.add_scalar("y=x*x",i*i,i)
    writer.close() 

#transform工具
def Usage3():
    image=Image.open("data/hymenoptera_data/ants_image/0013035.jpg")
    tensor_trans=transforms.ToTensor()
    tensor_image=tensor_trans(image)
    visualize_tensor_plotly(tensor_image,"origin")

    #标准化   
    trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])#标准差和方差
    norm_image=trans_norm(tensor_image)
    visualize_tensor_plotly(norm_image,"norm")

    #缩放
    resize_trans=transforms.Resize((500,500))
    resize_image=resize_trans(image)
    resize_image= tensor_trans(resize_image)
    visualize_tensor_plotly(resize_image,"resize")
    
    resize_trans_2=transforms.Resize(512)

if __name__ == "__main__":
    Usage3()




    
