import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class GANDataset(Dataset):
    "Dataset class for Kaggle GAN 2026 Competition"
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load all train images
        for class_idx in range(1, 13): # Class1 to Class12
            class_name = f"Class{class_idx}"
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
            
            for img_path in glob.glob(os.path.join(class_path, "*.jpg")):
                self.image_paths.append(img_path)
                self.labels.append(class_idx - 1) # 0-indexed labels for PyTorch

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Extract ID from filename (e.g., 101.jpg -> 101)
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_id

def get_dataloaders(train_dir, test_dir, batch_size=32, img_size=128):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_ds = GANDataset(train_dir, transform=train_transform)
    test_ds = TestDataset(test_dir, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

if __name__ == "__main__":
    train_dl, test_dl = get_dataloaders("train/train", "test_cases/images")
    print(f"Train batches: {len(train_dl)}, Test batches: {len(test_dl)}")
