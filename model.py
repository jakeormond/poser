import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler, SGD
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from create_dataset import CustomImageDataset
import matplotlib.pyplot as plt
import os
import datetime


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
    def __init__(self, n_outputs=8):
        super(CustomResNet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, n_outputs)
        self.relu = torch.nn.ReLU()

        # Add a series of deconvolutional layers
        self.deconv1 = torch.nn.ConvTranspose2d(n_outputs, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1)

        # Add a sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)

        # Reshape the output to have 4 dimensions
        x = x.view(x.shape[0], 16, int(x.shape[1]/16), int(x.shape[1]/16))

        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)

        # Apply the sigmoid activation function
        x = self.sigmoid(x)

        return x

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
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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
    
    history = {'train_loss': [], 'test_loss': []}

    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        print(f"train loss: {train_loss:>7f}")
        history['train'].append(train_loss)     

        test_loss = test_loop(test_dataloader, model, loss_fn, device)
        history['test'].append(test_loss)
        print(f"test loss: {test_loss:>7f}")
        step_lr_scheduler.step()

    return history


    pass
    


if __name__ == "__main__":

    
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

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=True)

    # Instantiate the model
    model = CustomResNet()
    model.to(device)

    loss_fn = nn.MSELoss()
    # optimizer = Adam(model.parameters(), lr=0.001)
    optimizer = SGD(model.parameters(), lr=0.001)

    # scheduler
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    # send a batch through and check output dimensions
    model.eval()
    inputs, _ = next(iter(train_dataloader))
    inputs = inputs.to(device)

    # Pass the batch through the model
    outputs = model(inputs)

    # Print the shape of the output
    print(outputs.shape)
    





    history = model_training(train_dataloader, test_dataloader, loss_fn, 
                   model, optimizer, device, step_lr_scheduler, n_epochs=10)

    # plot the training and test loss as subplots
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(history['train'])
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')

    ax[1].plot(history['test'])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Test Loss')

    plt.show()    
    
    # create a folder for the model in data_dir if one doesn't exist
    model_dir = os.path.join(data_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # create another folder in the models folder with the date and time
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join(model_dir, now)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # save the figure and the model
    fig_path = os.path.join(model_dir, 'losses.png')
    plt.savefig(fig_path)
    plt.close()          

    model_path = os.path.join(model_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)


    pass