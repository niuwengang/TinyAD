from torch.utils.data import Dataset
import cv2
import os 

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
    
def main():
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
    

if __name__ == "__main__":
    main()




    
