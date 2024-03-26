import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None, target_transform=None):

        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        image = image/255

        label = self.img_labels.iloc[idx, 1:]
        label = np.array(label).reshape(-1, 2)
        
        # create a empty set of heatmaps
        heatmap = torch.zeros((len(label), image.shape[1], image.shape[2]))

        # for each keypoint, create a heatmap and add it to the set of heatmaps
        for i, keypoint in enumerate(label):
            x, y = keypoint
            if np.isnan(x) or np.isnan(y):
                continue
            x = int(x)
            y = int(y)
            heatmap[i, x, y] = 1
      
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            heatmap = self.target_transform(heatmap)
        return image, heatmap
    
if __name__ == '__main__':

    img_dir = '/media/jake/LaCie/video_files/extracted_frames'
    annotations_file = '/media/jake/LaCie/video_files/labelled_frames/coordinates.csv'
    transform = transforms.Compose([transforms.Resize((350, 350))])  # Resize to 350x350
    target_transform = None

    annotations = pd.read_csv(annotations_file)

    # split the data into training, test, and validation sets
    n = len(annotations)

    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15

    train_labels, test_labels = train_test_split(annotations, test_size=1-train_ratio, shuffle=True)
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
    plt.show()

    # Display the label
    label = train_labels[0].squeeze()
    
    # plot the labels over the image
    plt.imshow(img, cmap="gray")
    for i in range(label.shape[0]):
        # get the x and y coordinates of the keypoint
        y, x = torch.where(label[i] == 1)
        x = int(x.numpy()/2)
        y = int(y.numpy()/2)
        # plot the keypoint
        plt.scatter(y, x, c='r', s=10)
        
    pass
    