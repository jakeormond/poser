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

def label_video(model, video_path, ds_size, output_path):

    # load video
    cap = cv2.VideoCapture(video_path)

    # properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # get the name of the video
    video_name = os.path.basename(video_path)
    # add labelled to the end of the video name
    labelled_video_name = video_name.split('.avi')[0] + '_labelled.mp4'
    labelled_video_path = os.path.join(output_path, labelled_video_name)

    video_writer = cv2.VideoWriter(labelled_video_path,
            cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), 
            fps, (height, width))

    while cap.isOpened():
        # read the next frame
        ret, frame = cap.read()        

        if ret:
            # get the frame size
            frame_size = frame.shape[:2]

            # downsample the frame
            frame_ds = cv2.resize(frame, (ds_size, ds_size))

            frame_ds = ToTensor(frame_ds)

            # Add an extra dimension for the batch size
            frame_ds = frame_ds.unsqueeze(0)

            # Pass the frame through the model
            input = frame_ds.to(device)
            output = model(input)

            # reshape output into 3d numpy array
            output = output.detach().cpu().numpy()
            output = output.squeeze(0)
            output = output.transpose((1,2,0))

            # process the output to get the labels
            labels = labels_from_heatmaps(output)

            # upsample the labels to the original frame size
            labels[:,0:2] = labels[:,0:2] * (frame_size[0] / ds_size)

            # label the frame
            frame = label_frame(frame, labels)

            # write the frame to the video
            video_writer.write(frame)
        
        else:
            # close the video writer
            video_writer.release()
            break
    
    pass


def labels_from_heatmaps(heatmaps):
    """
    Given a set of heatmaps, return the labels
    """
    n_labels = heatmaps.shape[2]
    labels = np.zeros((n_labels, 4))

    for i in range(n_labels):
        heatmap = heatmaps[:, :, i]
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        labels[i, 0:2] = [x, y]
        labels[i, 2] = heatmap[y, x]
        labels[i, 3] = np.sum(heatmap)

    return labels


def label_frame(frame, labels):
    """
    Given a frame and a set of labels, draw the labels on the frame
    """
    n_labels = labels.shape[0]
    for label in range(n_labels):
        x, y = labels[label, 0:2]
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # display the frame 
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     pass


    return frame


if __name__ == '__main__':
    
    initial_dir = '/media/jake/LaCie/video_files/models'    

    root = tk.Tk()
    root.withdraw()
    model_dir = filedialog.askdirectory(initialdir=initial_dir)

    # load history from pkl file
    # history_file = os.path.join(model_dir, 'history.pkl')
    # # load history from pkl file
    # with open(history_file, 'rb') as file:
    #     history = pickle.load(file)

    # plot_history(history, model_dir)

    ##### LABEL A VIDEO #####
    # create labelled_video directory in model_dir if it deosn't exist
    labelled_video_dir = os.path.join(model_dir, 'labelled_videos')
    if not os.path.exists(labelled_video_dir):
        os.makedirs(labelled_video_dir)

    # load model
    keypoints = ['tailbase']
    n_outputs = len(keypoints)
    model = CustomResNet(n_outputs)
    model_path = os.path.join(model_dir, 'model.pt')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # set downsampling size
    ds_size = 350
    
    # get the video file
    video_path = filedialog.askopenfilename(initialdir=initial_dir)

    # create labelled video directory if it doesn't already exist
    label_video(model, video_path, ds_size, labelled_video_dir)
    



    pass