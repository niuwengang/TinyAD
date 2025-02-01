from torch.utils.data import Dataset
from PIL import Image
import os 


class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.image_path=os.listdir(self.path)

    
    def __getitem__(self, idx):
        image_name=self.image_path[idx] 
        image_item_path=os.path.join(self.root_dir,self.label_dir,image_name)
        image=Image.open(image_item_path)
        label=self.label_dir
        return image,label
    
    def __len__(self):
        return len(self.image_path)
    
def main():
    root_dir="/home/g/workspace/TinyAD/data/hymenoptera_data/train"
    ants_label_dir="ants"
    bees_label_dir="bees"
    ants_dataset=MyData(root_dir,ants_label_dir)
    bees_dataset=MyData(root_dir,bees_label_dir)
    train_dataset=ants_dataset+bees_dataset
    print(len(train_dataset))
    image,label =train_dataset[120]
    image.show()

if __name__ == "__main__":
    main()




    
