# list all the frames in the directory
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global variable to store the coordinates
coords = []

# colormap is a global variable for consistency across functions
colormap = plt.get_cmap('viridis')

# This function will be called whenever the mouse is right-clicked
def draw_circle(event, x, y, flags, param):
    img = param[0]  # Get the image from the parameters
    if event == cv2.EVENT_LBUTTONUP:
        # Draw a circle where the user clicked
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(param[1], img)  # Update the image

        # Store the coordinates
        coords.append((x, y))

def scale_image(img, scale_factor):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img


def get_body_parts_from_df(df):
    # get list of body parts
    col_names = df.columns[1::2]
    # remove the _x suffix
    body_parts = [col.split('_')[0] for col in col_names]
    return body_parts


def label_frames(frame_names=None, dir=None, body_parts=None, relabel=False, scale_factor=2):

    if dir == None:
        dir = '/media/jake/LaCie/2023-05-03/videos/extracted_frames'

    # get the path to the parent directory
    parent_dir = os.path.dirname(dir)

    # create labelled frames directory in the parent directory if it doesn't exist
    labelled_frames_dir = os.path.join(parent_dir, 'labelled_frames')
    if not os.path.exists(labelled_frames_dir):
        os.makedirs(labelled_frames_dir)  

    csv_file = os.path.join(labelled_frames_dir, 'coordinates.csv')
    if os.path.exists(csv_file):
        print('The coordinates.csv file already exists in the directory')

        # load the csv file
        df = pd.read_csv(csv_file)
       

        if frame_names == None:
            # get the list of frames from the dataframe
            frame_names = df['frame'].values

        if body_parts == None:
            # get the list of body parts from the dataframe
            body_parts = get_body_parts_from_df(df)

    else: 
        # create and save the csv file
        if body_parts == None:
            # make a list of body parts to label on the animal
            body_parts = ['dot1', 'dot2', 'dot3', 'dot4', 'shoulder', 'spot1', 'spot2', 'tailbase']
                
        if frame_names == None:
            # find the png files
            frame_names = [f for f in os.listdir(dir) if f.endswith('.png')]      

        # create a dataframe to store the coordinates of the body parts
        # the first column will be the frame name, and the remaining columns 
        # will be the coordinates of the body parts
        df = pd.DataFrame(columns=['frame'] + [item for sublist in [[part + '_x', part + '_y'] for part in body_parts] for item in sublist])

        # enter the frame_names in the frame column of the dataframe
        df['frame'] = frame_names

        # save the dataframe to a csv file
        df.to_csv(csv_file, index=False)

    # get color list correspoding to each body part
    color_list_len = len(body_parts)
    color_list = [colormap(int(i)) for i in np.linspace(0, 255, color_list_len)]

    # loop through each frame, labelling each one
    # get the list of frames from the dataframe
    # frame_names = df['frame'].values
        
    for frame in frame_names:

        # get the index of the frame in the dataframe
        frame_index = df[df['frame'] == frame].index[0]

        if relabel==False:
            # check if there are any non-NaN values in the body part columns
            # if there are, skip the frame
            if not df.iloc[frame_index, 1:].isnull().all():
                continue

        # create the full path to the frames
        frame_path = os.path.join(dir, frame)

        # display the frame in a window
        img = cv2.imread(frame_path)

        # Scale the image
        # scale_factor = 2
        if scale_factor != 1:
            img = scale_image(img, scale_factor)
        
        # Create a named window
        cv2.namedWindow(frame)

        # Bind the function to window
        cv2.setMouseCallback(frame, draw_circle, param=(img, frame))   

        # Display the image
        cv2.imshow(frame, img)

        # loop through each body part
        n_body_parts = len(body_parts)
        j = 0
        while j < n_body_parts:

            body_part = body_parts[j]

            # prompt the user to click on the body part
            message = 'Click on the ' + body_part + ' of the animal'
            cv2.displayOverlay(frame, message, 10000)  # Display the message for 10 seconds

            # Clear the coordinates
            coords.clear()        

            while True:
                # wait for the user to click on the image
                k = cv2.waitKey(1) & 0xFF
                
                # # press ESC to exit the program
                # if k == 27:
                #     break

                # press "r" to re-select the last point
                if k == ord('r'):
                    
                    j -= 1

                    coords.clear()        

                    # Display the image
                    img = cv2.imread(frame_path)
                    if scale_factor != 1:
                        img = scale_image(img, scale_factor)
                    cv2.setMouseCallback(frame, draw_circle, param=(img, frame))   
                    cv2.imshow(frame, img)

                    # redraw the circles from 0 to i
                    for j2 in range(j):
                        column_name_x = body_parts[j2] + '_x'
                        column_name_y = body_parts[j2] + '_y'

                        # x is value in df at frame_index, column_name_x
                        if not np.isnan(df.loc[frame_index, column_name_x]):
                            x = (df.loc[frame_index, column_name_x] * scale_factor).astype(int)
                            y = (df.loc[frame_index, column_name_y] * scale_factor).astype(int)

                            # get color
                            color_temp = (np.array(color_list[j2][:-1]) * 255).astype(int)[::-1]
                            cv2.circle(img, (x, y), 5, color_temp.tolist(), -1)
                        
                        cv2.imshow(frame, img)        
                    
                    break

                # press "s" to skip the current body part
                if k == ord('s'):
                    j += 1
                    break

                # Print the coordinates
                if coords:

                    # adjust the coordinates to the original size of the image
                    coords[0] = (int(coords[0][0] / scale_factor), int(coords[0][1] / scale_factor))

                    # store the coordinates in the df              
                    df.loc[frame_index, body_part + '_x'] = int(coords[0][0])
                    df.loc[frame_index, body_part + '_y'] = int(coords[0][1])
                    j += 1
                    break
            
        
        # re-save the dataframe
        df.to_csv(csv_file, index=False)


        # Close all windows
        cv2.destroyAllWindows()

    return df

def remove_frames(frame_names=None, dir=None):
    
    if dir == None:
        dir = '/media/jake/LaCie/2023-05-03/videos'
    
    # extracted frames dir
    extracted_frames_dir = os.path.join(dir, 'extracted_frames')
    
    # labelled frames dir
    labelled_frames_dir = os.path.join(dir, 'labelled_frames')

    csv_file = os.path.join(labelled_frames_dir, 'coordinates.csv')
    # load the csv file
    df = pd.read_csv(csv_file)

    for frame in frame_names:
        # get the index of the frame in the dataframe
        frame_index = df[df['frame'] == frame].index[0]

        # remove that row from the dataframe
        df.drop(frame_index, inplace=True)

        # delete the png file from extracted_frames_dir
        frame_path = os.path.join(extracted_frames_dir, frame)
        os.remove(frame_path)

    # re-save the dataframe
    df.to_csv(csv_file, index=False)
    
    return df



def save_labelled_frames(dir):
    csv_file = os.path.join(dir, 'coordinates.csv')
    df = pd.read_csv(csv_file)

    # get list of body parts
    body_parts = get_body_parts_from_df(df)

    # get color list correspoding to each body part
    color_list_len = len(body_parts)
    # color list should transition from red to green to blue
    # first load a colormap
    # colormap = plt.get_cmap('viridis')
    # then take color_list_len colors from the colormap, equally spaced from beginning to end
    color_list = [colormap(int(i)) for i in np.linspace(0, 255, color_list_len)]

    # get parent dir
    parent_dir = os.path.dirname(dir)

    # extracted frames dir
    extracted_frames_dir = os.path.join(parent_dir, 'extracted_frames')

    # loop through the rows of df
    for i, row in df.iterrows():
        # get values in row series
        values = row.dropna().values[1:]
        # if all values are NaN, skip the row
        if len(values) == 0:
            continue        

        frame = row['frame']
        labelled_frame_path = os.path.join(dir, frame)
        # if labelled frame is already in the path, skip the frame
        if os.path.exists(labelled_frame_path):
            continue

        unlabelled_frame_path = os.path.join(extracted_frames_dir, frame)

        # display the frame in a window
        img = cv2.imread(unlabelled_frame_path)

        # loop through each body part
        for j, body_part in enumerate(body_parts):

            column_name_x = body_part + '_x'
            column_name_y = body_part + '_y'

            # x is value in df at frame_index, column_name_x
            x = row[column_name_x]
            y = row[column_name_y]

            if not np.isnan(x) and not np.isnan(y):
                color_temp = (np.array(color_list[j][:-1]) * 255).astype(int)[::-1]
                cv2.circle(img, (int(x), int(y)), 5, color_temp.tolist(), -1)

        # save the image
        cv2.imwrite(labelled_frame_path, img)

    pass

if __name__ == '__main__':
    video_dir = '/media/jake/LaCie/video_files'
    extracted_frames_dir = os.path.join(video_dir, 'extracted_frames')
    labelled_frames_dir = os.path.join(video_dir, 'labelled_frames')
    body_parts = ['dot1', 'dot2', 'dot3', 'dot4', 'shoulder', 'spot1', 'spot2', 'tailbase']

    # load the csv file
    # csv_file = os.path.join(labelled_frames_dir, 'coordinates.csv')
    # df = pd.read_csv(csv_file)
    # frame_names=['video_2024-02-17_10.16.56_frame_1539.png']
    # null_frames=['video_2024-02-17_10.03.41_frame_0.png']
    
    # remove_frames(frame_names=null_frames, dir=video_dir)
    
    frames = ['video_2024-02-17_09.38.45_frame_852.png']
    # df = label_frames(frame_names=frames, dir=extracted_frames_dir, 
    #                 body_parts=body_parts, relabel=True, scale_factor=1.8)

    df = label_frames(dir=extracted_frames_dir, body_parts=body_parts, scale_factor=1.8)


    # csv_file = os.path.join(labelled_frames_dir, 'coordinates.csv')
    # df = pd.read_csv(csv_file)

    save_labelled_frames(labelled_frames_dir)

    pass        
        


        