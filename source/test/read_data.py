from torch.utils.data import Dataset
import cv2

class MyData(Dataset):
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)