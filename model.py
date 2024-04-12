import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler, SGD
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

from create_dataset import CustomImageDataset
from loss_fn import JointsMSELoss

import matplotlib.pyplot as plt
import os
import datetime

import tkinter as tk
from tkinter import filedialog

import pickle

import numpy as np


device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

# model = resnet50(weights=ResNet50_Weights.DEFAULT)
# num_ftrs = model.fc.in_features

# n_outputs = 16
# model.fc = torch.nn.Linear(num_ftrs, n_outputs)

# # add a relu layer
# model.fc.add_module('relu', torch.nn.ReLU())

class CustomResNet(torch.nn.Module):
    def __init__(self, n_outputs):
        super(CustomResNet, self).__init__()
        # full_resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        full_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # full_resnet = resnet50(weights=None)
        self.resnet = torch.nn.Sequential(*(list(full_resnet.children())[:-2]))
        # num_ftrs = self.resnet.fc.in_features
        # self.resnet.fc = torch.nn.Linear(num_ftrs, n_outputs)       
        self.relu = torch.nn.ReLU()

        # get number of outputs from the resnet model
        # Get the last convolutional layer in the Sequential object
        # last_conv_layer = next(layer for layer in reversed(list(self.resnet.children())) 
        #                        if isinstance(layer, torch.nn.modules.conv.Conv2d))

        # Get the number of output channels
        # n_outputs = last_conv_layer.out_channels


        # Add a series of deconvolutional layers with BatchNorm and ReLU activations
        
        self.deconv1 = torch.nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
        # self.deconv1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        
        
        self.deconv2plus = torch.nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv_final = torch.nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=2)

        self.batchnorm = torch.nn.BatchNorm2d(256)

        self.final_conv = torch.nn.Conv2d(256, n_outputs, kernel_size=1, stride=1, padding=0)

        # Add a sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

        # softmax activation
        self.softmax = torch.nn.Softmax2d()

    def forward(self, x):
        x = self.resnet(x)
        # print the shape of the output of the resnet model
        # print(f'x shape after resnet: {x.shape}')


        # x = self.relu(x)

        # Apply the deconvolutional layers
        x = self.relu(self.batchnorm(self.deconv1(x)))       
        x = self.relu(self.batchnorm(self.deconv2plus(x)))
        x = self.relu(self.batchnorm(self.deconv2plus(x)))
        x = self.relu(self.batchnorm(self.deconv2plus(x)))
        x = self.relu(self.batchnorm(self.deconv_final(x)))

        # Apply the final convolutional layer
        x = self.final_conv(x)

        # apply a final softmax layer
        # x = self.softmax(x)

        # Apply the sigmoid activation function
        # x = self.sigmoid(x)

        # normalize x so that the sum of each heatmap is 1
        # x = x / x.sum(dim=[2, 3], keepdim=True)

        return x
    

    def save(self, optimizer, model_dir):
        # if model dir doesn't exist, create it
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # create another folder in the models folder with the date and time
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = os.path.join(model_dir, now)
        self.model_dir = model_dir

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)  

        # save the model
        model_path = os.path.join(model_dir, 'model.pt')
        torch.save(self.state_dict(), model_path)

        # save the optimizer
        optimizer_path = os.path.join(model_dir, 'optimizer.pt')
        torch.save(optimizer.state_dict(), optimizer_path)

        return model_dir
    

    def dir(self):
        return self.model_dir

# train and test loops
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    history = []
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device))

        # plt.imshow(np.transpose(y[0].detach().cpu().numpy(), (1, 2, 0)))
        # plt.imshow(np.transpose(pred[0].detach().cpu().numpy(), (1, 2, 0)))

        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * batch_size + len(X)

        history.append(loss)

        if batch % 10 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")   

    return history    


def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_loss = 0
    num_batches = len(dataloader)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    return test_loss

def model_training(train_dataloader, test_dataloader, loss_fn, 
                   model, optimizer, device, step_lr_scheduler, n_epochs=10):
    
    # set model to training mode
    model.train()
    history = {'train_loss': [], 'test_loss': []}

    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        print(f"train loss: {train_loss[-1]:>7f}")
        history['train_loss'].append(train_loss)     

        test_loss = test_loop(test_dataloader, model, loss_fn, device)
        history['test_loss'].append(test_loss)
        print(f"test loss: {test_loss:>7f}")
        step_lr_scheduler.step()

    return history


def check_output_dimensions(model, dataloader):
    # send a batch through and check output dimensions
    model.eval()
    inputs, labels = next(iter(dataloader))

    print(f'inputs shape: {inputs.shape}')
    print(f'labels shape: {labels.shape}')

    inputs = inputs.to(device)

    # Pass the batch through the model
    outputs = model(inputs)
    print(f'outputs shape: {outputs.shape}')

    return 


def plot_history(history, model_dir=None):

    train_loss = []
    test_loss = []

    for i in range(len(history)):
        for e in range(len(history[i]['test_loss'])):
            test_loss.append(history[i]['test_loss'][e])

            train_loss_temp = []
            for b in range(len(history[i]['train_loss'][e])):
                train_loss_temp.append(np.mean(history[i]['train_loss'][e][b]))
        
            train_loss.append(np.mean(train_loss_temp))
    
    # create figure with 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    # add title
    fig.suptitle('Training and Test Losses')
    
    # plot the training and test losses in first subplot
    ax[0].plot(train_loss, label='train loss')
    ax[0].plot(test_loss, label='test loss')

    # add labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # in second subplot, plot from the 201 epoch on
    ax[1].plot(train_loss[200:], label='train loss')
    ax[1].plot(test_loss[200:], label='test loss')

    # add labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()  

    if model_dir is not None:
        fig_path = os.path.join(model_dir, 'losses.png')
        plt.savefig(fig_path)  

    return





def get_dataloaders(annotations_file, transform=None, labels=None, keypoints=None, sigma=None, batch_size=8):

    # transform = transforms.Compose([transforms.Resize((ds_size, ds_size))])  

    if labels is None:
        annotations = pd.read_csv(annotations_file)

        # split the data into training, test, and validation sets

        train_ratio = 0.7
        validation_ratio = 0.15
        test_ratio = 0.15

        train_labels, test_labels = train_test_split(annotations, test_size=1-train_ratio, shuffle=True)
        val_labels, test_labels = train_test_split(test_labels, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=True)

    else:
        train_labels = labels['train']
        test_labels = labels['test']
        val_labels = labels['validation']
    
    if sigma is None:   
        sigma = 100
    
    training_data = CustomImageDataset(train_labels, img_dir, keypoints=keypoints, sigma=sigma, transform=transform)
    test_data = CustomImageDataset(test_labels, img_dir, keypoints=keypoints, sigma=sigma, transform=transform)
    validation_data = CustomImageDataset(val_labels, img_dir, keypoints=keypoints, sigma=sigma, transform=transform)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['test'] = test_dataloader
    dataloaders['validation'] = validation_dataloader

    labels = {}
    labels['train'] = train_labels
    labels['test'] = test_labels
    labels['validation'] = val_labels
    return dataloaders, labels


if __name__ == "__main__":

    load_model = False

    data_dir = '/media/jake/LaCie/video_files'

    # if model dir doesn't exist, create it
    model_dir = os.path.join(data_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # create another folder in the models folder with the date and time
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_dir_now = os.path.join(model_dir, now)

    img_dir = '/media/jake/LaCie/video_files/extracted_frames'
    annotations_file = '/media/jake/LaCie/video_files/labelled_frames/coordinates.csv'

    keypoints = ['tailbase']
    
    
    ds_size = 350

    # run groups of epochs with decresing sigmas
    # sigmas = [100, 85, 70, 60, 50]
    # sigmas = [20, 18, 15, 12, 9]

    # sigmas = [100, 85, 70, 60, 50, 20, 18, 15, 12, 9]
    # sigmas = [8, 7, 6, 5, 4, 3, 2, 1]
    sigmas = [50]

    ds_size = 350
    transform = transforms_v2.Compose([
        transforms_v2.Resize((ds_size, ds_size)),
        transforms_v2.RandomHorizontalFlip(),
        transforms_v2.RandomRotation(10),
        transforms_v2.RandomResizedCrop(ds_size),
        transforms_v2.ToTensor()
    ])

    # load or initialize the model
    n_outputs = len(keypoints)
    model = CustomResNet(n_outputs)
    model.to(device)

    # optimizer = Adam(model.parameters(), lr=0.001)
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
    # optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
   
    history = []

    if load_model:
        # 
        import tkinter as tk
        from tkinter import filedialog

        model_dir_now = filedialog.askdirectory(initialdir=model_dir)
        model_path = os.path.join(model_dir_now, 'model.pt')
        model.load_state_dict(torch.load(model_path))
        
        optimizer_path = os.path.join(model_dir_now, 'optimizer.pt')
        optimizer.load_state_dict(torch.load(optimizer_path))

        # load the labels dictionary as a pickle file
        labels_path = os.path.join(model_dir_now, 'labels.pkl')
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)

        # load the history as a pickle file
        history_path = os.path.join(model_dir_now, 'history.pkl')
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        # plot the history
        plot_history(history, model_dir_now)


    # loss_fn = nn.MSELoss()
    loss_fn =  JointsMSELoss()

    
    # optimizer = SGD(model.parameters(), lr=0.001)
    # scheduler
    # step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=.99)
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, [50], gamma=.1)
    n_epochs = 200
    for i, sigma in enumerate(sigmas):
        if i == 0 and not load_model: 
            dataloaders, labels = get_dataloaders(annotations_file, transform=transform, keypoints=keypoints, sigma=sigma)

        else:
            dataloaders, _ = get_dataloaders(annotations_file, transform=transform, keypoints=keypoints, labels=labels, sigma=sigma)
            
        train_dataloader = dataloaders['train']
        test_dataloader = dataloaders['test']
        validation_dataloader = dataloaders['validation']        
        
        ######################### TRAIN THE MODEL ################################### 
        # if sigma > 60:
        #     n_epochs = 30

        # else:
        #     n_epochs = 15   

        history.append(model_training(train_dataloader, test_dataloader, loss_fn, 
                    model, optimizer, device, step_lr_scheduler, n_epochs=n_epochs))  
    
    
    ########### SAVE THE MODEL, OPTIMIZER, AND LABELS ############################
    model_dir = model.save(optimizer, model_dir)   
    # save the labels dictionary as a pickle file
    labels_path = os.path.join(model_dir, 'labels.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f)

    # save the history
    history_path = os.path.join(model_dir, 'history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    plot_history(history, model.dir())


    ############################## LOAD THE MODEL #################################
    # load the trained model from the model.pt file in model_dir
    model = CustomResNet()

    # choose the directory using gui within the model_dir
    model_dir_now = filedialog.askdirectory(initialdir=model_dir)
    model_path = os.path.join(model_dir_now, 'model.pt')
    model.load_state_dict(torch.load(model_path))    
    model.to(device)
    
    ########### PLOT SOME TRAINING EXAMPLES ##################
    model.eval()
    inputs, labels = next(iter(train_dataloader))
    inputs = inputs.to(device)

    # Pass the batch through the model
    outputs = model(inputs)

    # make a figure with 2 square subplot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(inputs[0].cpu().detach().numpy().transpose(1,2,0))

    # plot the labels
    for i in range(labels.shape[1]):
        y, x = torch.where(labels[0,i,:,:] == labels[0,i,:,:].max())
        x = x.numpy()
        y = y.numpy()
        ax[0].scatter(y, x, c='r', s=10)
        
    # plot the outputs
    ax[1].imshow(inputs[0].cpu().detach().numpy().transpose(1,2,0))
    outputs = outputs.cpu().detach()
    for i in range(outputs.shape[1]):
        y, x = torch.where(outputs[0,i,:,:] == outputs[0,i,:,:].max())
        x = x.numpy()
        y = y.numpy()
        ax[1].scatter(y, x, c='b', s=10)



    # check trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    pass