import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import torchvision.transforms.v2 as transforms

# We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
# some APIs may slightly change in the future
import torchvision
torchvision.disable_beta_transforms_warning()


from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, annotations, img_dir, keypoints=None, sigma=None, transform=None):

        self.img_labels = annotations

        if  keypoints is None:
            # get column names
            keypoints_xy = self.img_labels.columns[1:]
            # keep only the names that end with _x and remove the _x
            self.keypoints = [name[:-2] for name in keypoints_xy if name[-2:] == '_x']
        # keep only the names that are in the keypoints list
        else:
            self.keypoints = keypoints

        self.img_dir = img_dir
        self.transform = transform

        if sigma is None:
            self.sigma = 100
        else:
            self.sigma = sigma

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        image = image/255

        # fig = plt.figure()
        # image_for_plot = image.permute(1, 2, 0)
        # plt.imshow(image_for_plot)

        label_ind =[]
        for keypoint in self.keypoints:
            label_ind.append(self.img_labels.columns.get_loc(keypoint + '_x'))
            label_ind.append(self.img_labels.columns.get_loc(keypoint + '_y'))

        
        label = self.img_labels.iloc[idx, label_ind]
        label = np.array(label).reshape(-1, 2)
        
        # create an empty set of heatmaps
        heatmap = torch.zeros((len(label), image.shape[1], image.shape[2]))

        # for each keypoint, create a heatmap and add it to the set of heatmaps
        for i, keypoint in enumerate(label):
            x, y = keypoint
            if np.isnan(x) or np.isnan(y):
                continue
            x = int(x)
            y = int(y) 
            heatmap[i,:,:] = apply_gaussian(heatmap[i,:,:], y, x, self.sigma)

      
        if self.transform:    
            # concatenate image and heatmap, transform, and then split them
            image = torch.cat((image, heatmap), dim=0)
            image = self.transform(image)
            image, heatmap = torch.split(image, [3, len(label)], dim=0)
                    
            # just for testing
            # keypoints_ds = get_keypoints(heatmap)

            # normalize the heatmaps
            heatmap = normalize_heatmap(heatmap)

            # plot image and heatmaps
            # fig, ax = plt.subplots(1,2, figsize=(10,5))
            # image_for_plot = image.permute(1, 2, 0)
            # ax[0].imshow(image_for_plot)

            # for i in range(heatmap.shape[0]):
            #     # get the x and y coordinates of the keypoint
            #     y, x = torch.where(heatmap[i] == 1)
            #     if x.numel() == 0:
            #         continue
            #     x = int(x.numpy())
            #     y = int(y.numpy())
            #     # plot the keypoint
            #     ax[0].scatter(x, y, c='r', s=10)

            # # sum the heatmaps to get a single heatmap
            # heatmap_sum = heatmap.sum(dim=0)
            # ax[1].imshow(heatmap_sum)

        return image, heatmap
    

class DatasetForVideoLabelling(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = image/255

        # fig = plt.figure()
        # image_for_plot = image.permute(1, 2, 0)
        # plt.imshow(image_for_plot)
        
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
            heatmap[i,:,:] = apply_gaussian(heatmap[i,:,:], y, x, self.sigma)

      
        if self.transform:    
            # concatenate image and heatmap, transform, and then split them
            image = torch.cat((image, heatmap), dim=0)
            image = self.transform(image)
            image, heatmap = torch.split(image, [3, len(label)], dim=0)
                    
            # just for testing
            # keypoints_ds = get_keypoints(heatmap)

            # normalize the heatmaps
            heatmap = normalize_heatmap(heatmap)

            # plot image and heatmaps
            # fig, ax = plt.subplots(1,2, figsize=(10,5))
            # image_for_plot = image.permute(1, 2, 0)
            # ax[0].imshow(image_for_plot)

            # for i in range(heatmap.shape[0]):
            #     # get the x and y coordinates of the keypoint
            #     y, x = torch.where(heatmap[i] == 1)
            #     if x.numel() == 0:
            #         continue
            #     x = int(x.numpy())
            #     y = int(y.numpy())
            #     # plot the keypoint
            #     ax[0].scatter(x, y, c='r', s=10)

            # # sum the heatmaps to get a single heatmap
            # heatmap_sum = heatmap.sum(dim=0)
            # ax[1].imshow(heatmap_sum)

        return image, heatmap
 

def apply_gaussian(heatmap, y, x, sigma):
    grid_y, grid_x = np.ogrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
    gaussian = np.exp(-((grid_x-x)**2 + (grid_y-y)**2) / (2.*sigma**2))
    heatmap += gaussian
    return heatmap

  
def normalize_heatmap(heatmap):
    # make the max value in each heatmap 1
    for i in range(heatmap.shape[0]):
        if heatmap[i].max() != 0:
            heatmap[i] = 2*(heatmap[i]/heatmap[i].max())
    return heatmap
    
if __name__ == '__main__':

    img_dir = '/media/jake/LaCie/video_files/extracted_frames'
    annotations_file = '/media/jake/LaCie/video_files/labelled_frames/coordinates.csv'
    transform = transforms.Compose([transforms.Resize((350, 350))])  # Resize to 350x350
    target_transform = transforms.Compose([transforms.Resize((350, 350))])

    annotations = pd.read_csv(annotations_file)

    # split the data into training, test, and validation sets
    n = len(annotations)

    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15

    train_labels, test_labels = train_test_split(annotations, test_size=1-train_ratio, shuffle=True)
    val_labels, test_labels = train_test_split(test_labels, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=True)


    training_data = CustomImageDataset(train_labels, img_dir, keypoints=['tailbase'], transform=transform)
    test_data = CustomImageDataset(test_labels, img_dir, keypoints=['tailbase'], transform=transform)
    validation_data = CustomImageDataset(val_labels, img_dir, keypoints=['tailbase'], transform=transform)

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
    # transpose the label
    label = label.permute(0,2,1)
    
    # plot the labels over the image
    plt.imshow(img, cmap="gray")
    for i in range(label.shape[0]):
        # get the x and y coordinates of the keypoint
        y, x = torch.where(label[i] == 1)
        x = int(x.numpy()/2)
        y = int(y.numpy()/2)
        # plot the keypoint
        plt.scatter(x, y, c='r', s=10)
        
    pass
    