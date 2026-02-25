import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class MultiLabelDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Read labels with space delimiter. 
        # Columns: ImageName, Attr1, Attr2, Attr3, Attr4
        self.df = pd.read_csv(labels_file, sep=' ', header=None)
        self.df.columns = ['img_name', 'attr1', 'attr2', 'attr3', 'attr4']
        
        # Mapping NA to -1, 0 to 0, 1 to 1
        # Pandas might read 'NA' as actual NaN floats. We handle both.
        self.df = self.df.replace('NA', -1).fillna(-1)
        self.df.iloc[:, 1:] = self.df.iloc[:, 1:].astype(float)

        
        # Filter out missing images
        existing_indices = []
        for i, img_name in enumerate(self.df['img_name']):
            if os.path.exists(os.path.join(self.img_dir, img_name)):
                existing_indices.append(i)
            else:
                print(f"Warning: Image {img_name} not found. Skipping.")
        
        self.df = self.df.iloc[existing_indices].reset_index(drop=True)
        print(f"Dataset initialized with {len(self.df)} images.")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        # Get labels as a float tensor
        labels = self.df.iloc[idx, 1:].values.astype('float32')
        labels = torch.tensor(labels)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

if __name__ == '__main__':
    # Simple test
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = MultiLabelDataset('labels.txt', 'images', transform=transform)
    img, labels = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Labels: {labels}")
