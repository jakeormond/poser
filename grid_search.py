import os
import pickle
import numpy as np
import json
import torch
from torch.optim import Adam, lr_scheduler, SGD
import torchvision.transforms.v2 as transforms_v2
from sklearn.model_selection import ParameterGrid

from model import CustomResNet
from loss_fn import JointsMSELoss
from create_dataset import get_dataloaders
from model import model_training, create_curr_model_dir, plot_history


device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

ds_size = 350 # set downsampling size
transform = transforms_v2.Compose([
        transforms_v2.Resize((ds_size, ds_size)),
        transforms_v2.RandomHorizontalFlip(),
        transforms_v2.RandomRotation(10),
        transforms_v2.RandomResizedCrop(ds_size),
        transforms_v2.ToTensor()
    ])


if __name__ == "__main__":

    data_dir = '/media/jake/LaCie/video_files'
    img_dir = '/media/jake/LaCie/video_files/extracted_frames'
    annotations_file = '/media/jake/LaCie/video_files/labelled_frames/coordinates.csv'
    keypoints = ['tailbase']
    
    # get number of inputs
    n_outputs = len(keypoints)
    loss_fn =  JointsMSELoss()
    batch_size = 8
    n_epochs = 20
   
    # if model dir doesn't exist, create it
    model_dir = os.path.join(data_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define the hyperparameters to tune
    param_grid = {
        'learning_rate': [0.05, 0.01, 0.001, 0.0001],
        'sigma': [100, 85, 70, 60, 50, 40, 30, 20, 10],
        'momentum': [0.9, 0.95, 0.99],
        'weight_decay': [0.001, 0.0001, 0.00001],
        # Add other hyperparameters here
    }

    # Create a grid of hyperparameter combinations
    param_combinations = list(ParameterGrid(param_grid))    

    # get the number of combinations
    n_combinations = len(param_combinations)
    # make a list of 20 random integers between 0 and n_combinations with no repeats, sorted
    random_indices = np.sort(np.random.choice(n_combinations, 20, replace=False))

    for i, ind in enumerate(random_indices):
        # get the parameters
        params = param_combinations[ind]

        # get dataloaders
        if i == 0:
            dataloaders, labels = get_dataloaders(annotations_file, img_dir, transform=transform, \
                                keypoints=keypoints, sigma=params['sigma'], batch_size=8)
        else:
            dataloaders, _ = get_dataloaders(annotations_file, img_dir, transform=transform, \
                keypoints=keypoints, sigma=params['sigma'], batch_size=8, labels=labels)
            
        train_dataloader = dataloaders['train']
        test_dataloader = dataloaders['test']
        validation_dataloader = dataloaders['validation']       

        # Create a new model
        model = CustomResNet(n_outputs)
        model.to(device)
        # Create a new optimizer with the current learning rate
        optimizer = SGD(model.parameters(), lr=params['learning_rate'], \
            momentum=params['momentum'], weight_decay=params['weight_decay'])

        history = model_training(train_dataloader, test_dataloader, loss_fn, 
                    model, optimizer, device, n_epochs=n_epochs)
        
        # Save the model and history
        model_dir_now = model.save(optimizer, model_dir)   
        # save the labels dictionary as a pickle file
        labels_path = os.path.join(model_dir_now, 'labels.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f)

        # save the history
        history_path = os.path.join(model_dir_now, 'history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

        plot_history(history, model.dir())

        # Save the hyperparameters
        params_path = os.path.join(model_dir_now, 'params.json')
        # add n_epochs to the params dictionary
        params['n_epochs'] = n_epochs
        params['optimizer'] = 'SGD'

        with open(params_path, 'w') as f:
            json.dump(params, f)