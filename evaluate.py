import tkinter as tk
from tkinter import filedialog
import os
import pickle

import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import to_tensor as ToTensor

from model import plot_history, CustomResNet

import cv2
import numpy as np

device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

def label_video(model, video_path, ds_size, output_path, label_path):

    # load video
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # read the next frame
        ret, frame = cap.read()

        # get the frame size
        frame_size = frame.shape[:2]

        # downsample the frame
        frame_ds = cv2.resize(frame, (ds_size, ds_size))

        if ret:
            frame_ds = ToTensor()(frame)

            # Add an extra dimension for the batch size
            frame_ds = frame_ds.unsqueeze(0)

            # Pass the frame through the model
            output = model(frame)

            # process the output to get the labels
            labels = labels_from_heatmaps(output)

            # upsample the labels to the original frame size
            labels = labels * (frame_size / ds_size)






    
    pass
    return labels


def labels_from_heatmaps(heatmaps):
    """
    Given a set of heatmaps, return the labels
    """
    n_labels = heatmaps.shape[1]
    labels = np.zeros((n_labels, 2))

    for i in range(n_labels):
        heatmap = heatmaps[:, i, :, :]
        x, y = np.unravel_index(heatmap.argmax(), heatmap.shape)
        labels[i, :] = [x, y]

    return labels


def label_frame(frame, labels):
    """
    Given a frame and a set of labels, draw the labels on the frame
    """
    for label in labels:
        x, y = label
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    return frame


if __name__ == '__main__':
    
    initial_dir = '/media/jake/LaCie/video_files/models'    

    root = tk.Tk()
    root.withdraw()
    model_dir = filedialog.askdirectory(initialdir=initial_dir)

    # load history from pkl file
    history_file = os.path.join(model_dir, 'history.pkl')
    # load history from pkl file
    with open(history_file, 'rb') as file:
        history = pickle.load(file)

    plot_history(history, model_dir)

    ##### LABEL A VIDEO #####
    # create labelled_video directory in model_dir if it deosn't exist
    labelled_video_dir = os.path.join(model_dir, 'labelled_videos')
    if not os.path.exists(labelled_video_dir):
        os.makedirs(labelled_video_dir)

    # load model
    model = CustomResNet()
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # set downsampling size
    ds_size = 350
    
    # get the video file
    video_path = filedialog.askopenfilename(initialdir=initial_dir)


    
    



    pass