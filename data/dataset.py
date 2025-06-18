import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter

class BreakHisDataset(Dataset):
    def __init__(self, data_path, magnification='40X', train=True, test_split=0.2, seed=42, transform=None):
        self.data_path = os.path.abspath(data_path)
        self.transform = transform
        self.train = train
        self.magnification = magnification

        random.seed(seed)

        if magnification == 'all':
            mags = ['40X', '100X', '200X', '400X']
        else:
            mags = [magnification]

        benign_files = []
        malignant_files = []

        for mag in mags:
            print(f"\n🔍 處理倍率：{mag}")

            # 良性圖像路徑
            benign_pattern = os.path.join(self.data_path, "breast", "benign", "SOB", "*", "*", mag, "*.*")
            b = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
                b.extend(glob.glob(benign_pattern.replace("*.*", ext), recursive=True))
            print(f"  ✅ 良性圖數量：{len(b)}")
            benign_files.extend(b)

            # 惡性圖像路徑
            malignant_pattern = os.path.join(self.data_path, "breast", "malignant", "SOB", "*", "*", mag, "*.*")
            m = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
                m.extend(glob.glob(malignant_pattern.replace("*.*", ext), recursive=True))
            print(f"  ✅ 惡性圖數量：{len(m)}")
            malignant_files.extend(m)

        if len(benign_files) == 0 and len(malignant_files) == 0:
            raise FileNotFoundError(f"在路徑 {self.data_path} 中未找到任何圖像文件")

        benign_labels = [0] * len(benign_files)
        malignant_labels = [1] * len(malignant_files)

        all_files = benign_files + malignant_files
        all_labels = benign_labels + malignant_labels

        train_files, test_files, train_labels, test_labels = train_test_split(
            all_files, all_labels, test_size=test_split, random_state=seed, stratify=all_labels
        )

        if train:
            self.image_files = train_files
            self.labels = train_labels
        else:
            self.image_files = test_files
            self.labels = test_labels

        print(f"\n📦 {'訓練集' if train else '測試集'}總數量: {len(self.image_files)} 張")
        print("📊 標籤分佈:", Counter(self.labels))  # 顯示 0: xxx, 1: yyy

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ 圖片載入錯誤 {img_path}: {e}")
            if idx != 0 and len(self.image_files) > 0:
                return self.__getitem__(0)
            else:
                image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(data_path, batch_size=32, magnification='40X', test_split=0.2, seed=42, num_workers=4):
    data_path = os.path.abspath(data_path)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BreakHisDataset(
        data_path=data_path,
        magnification=magnification,
        train=True,
        test_split=test_split,
        seed=seed,
        transform=train_transform
    )

    test_dataset = BreakHisDataset(
        data_path=data_path,
        magnification=magnification,
        train=False,
        test_split=test_split,
        seed=seed,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_class_distribution(data_path, magnification='40X'):
    data_path = os.path.abspath(data_path)

    benign_types = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
    malignant_types = ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]

    distribution = {}

    mags = ['40X', '100X', '200X', '400X'] if magnification == 'all' else [magnification]

    for b_type in benign_types:
        count = 0
        for mag in mags:
            pattern = os.path.join(data_path, "benign", "SOB", b_type, "*", mag, "*.png")
            files = glob.glob(pattern, recursive=True)
            if len(files) == 0:
                for ext in ['*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
                    files.extend(glob.glob(pattern.replace("*.png", ext), recursive=True))
            count += len(files)
        distribution[f"benign/{b_type}"] = count

    for m_type in malignant_types:
        count = 0
        for mag in mags:
            pattern = os.path.join(data_path, "**", "malignant", "**", m_type, "**", mag, "*.png")
            files = glob.glob(pattern, recursive=True)
            if len(files) == 0:
                for ext in ['*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
                    files.extend(glob.glob(pattern.replace("*.png", ext), recursive=True))
            count += len(files)
        distribution[f"malignant/{m_type}"] = count

    distribution["benign_total"] = sum(distribution[f"benign/{b_type}"] for b_type in benign_types)
    distribution["malignant_total"] = sum(distribution[f"malignant/{m_type}"] for m_type in malignant_types)
    distribution["total"] = distribution["benign_total"] + distribution["malignant_total"]

    return distribution
