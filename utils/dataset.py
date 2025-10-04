from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# image net transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

import os
class HoaVNDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.data, self.label = self.load_images(root_path)
    
    
    def convert_label(self, label):
        label_dict = {
            'Cuc' : 0,
            'Dao' : 1,
            'Lan' : 2,
            'Mai' : 3,
            'Tho' : 4
        }
        return label_dict[label]
    def load_images(self, root_path):
        imgs_path = []
        angel_label = []
        for label in os.listdir(root_path):
            label_path = os.path.join(root_path, label)
            for img in os.listdir(label_path):
                img_path = os.path.join(label_path, img)
                img = Image.open(img_path).convert("RGB")
                
                imgs_path.append(img)
                angel_label.append(self.convert_label(label))

        return imgs_path, angel_label

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img =  self.data[idx]
        label = self.label[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

import logging
import os 
def logg(log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )