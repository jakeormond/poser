import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, labels, img_dir, frame_size=700, transform=None, target_transform=None):

        self.img_labels = labels.iloc[:, 1:].to_numpy()
        self.img_names = labels.frame
        self.img_dir = img_dir
        self.frame_size = frame_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names.iloc[idx])
        image = read_image(img_path)
        image = image/255
        label = self.img_labels[idx, :]
        label = torch.tensor(label/self.frame_size)
        label = label.float()
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
if __name__ == '__main__':

    img_dir = '/media/jake/LaCie/2023-05-03/videos/extracted_frames'
    annotations_file = '/media/jake/LaCie/2023-05-03/videos/extracted_frames/coordinates.csv'
    transform = None
    target_transform = None

    labels = pd.read_csv(annotations_file)

    # split the data into training, test, and validation sets
    n = len(labels)

    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15

    train_labels, test_labels = train_test_split(labels, test_size=1-train_ratio, shuffle=True)
    val_labels, test_labels = train_test_split(test_labels, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=True)


    training_data = CustomImageDataset(train_labels, img_dir, transform=transform, target_transform=target_transform)
    test_data = CustomImageDataset(test_labels, img_dir, transform=transform, target_transform=target_transform)
    validation_data = CustomImageDataset(val_labels, img_dir, transform=transform, target_transform=target_transform)

    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=16, shuffle=True)

    # Display an image and label
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    # reshape image to so first dimension becomes third
    img = img.permute(1, 2, 0)
    img = img * 255    
    img = img.numpy().astype('uint8')
    # convert to unint8
    #img = img.astype('uint8')
    plt.imshow(img, cmap="gray")
    plt.imshow(img)
    plt.show()

    label = train_labels[0]
    print(f"Label: {label}")

    pass
    