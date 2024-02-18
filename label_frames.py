# list all the frames in the directory
import os
import cv2
import numpy as np
import pandas as pd


# Global variable to store the coordinates
coords = []


# This function will be called whenever the mouse is right-clicked
def draw_circle(event, x, y, flags, param):
    img = param  # Get the image from the parameters
    if event == cv2.EVENT_LBUTTONUP:
        # Draw a circle where the user clicked
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow('image', img)  # Update the image

        # Store the coordinates
        coords.append((x, y))

def scale_image(img, scale_factor):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img


def label_frames(dir=None, body_parts=None):

    if dir == None:
        dir = '/media/jake/LaCie/2023-05-03/videos/extracted_frames'

    if body_parts == None:
        # make a list of body parts to label on the animal
        body_parts = ['dot1', 'dot2', 'dot3', 'dot4', 'shoulder', 'spot1', 'spot2', 'tail_base']

    # make a new directory to store the labeled frames
    if not os.path.exists(os.path.join(dir, 'labeled_frames')):
        os.makedirs(os.path.join(dir, 'labeled_frames'))
        
    # find the png files
    frame_names = [f for f in os.listdir(dir) if f.endswith('.png')]
    # create the full paths to the frames
    frames = [os.path.join(dir, f) for f in frame_names]

    # create an empty array that has the same number of rows as the number of frames, 
    # and the same number of columns as twice the number of body parts (for x and y 
    # coordinates); this will be used to store the coordinates of the body parts
    coordinates = np.zeros((len(frames), len(body_parts)*2))

    # loop through each frame
    for frame in frames:
        # display the frame in a window
        img = cv2.imread(frame)

        # Scale the image
        scale_factor = 2
        img = scale_image(img, scale_factor)
        
        # Create a named window
        cv2.namedWindow('image')

        # Bind the function to window
        cv2.setMouseCallback('image', draw_circle, img)   

        # Display the image
        cv2.imshow('image', img)

        # loop through each body part
        n_body_parts = len(body_parts)
        i = 0
        while i < n_body_parts:

            body_part = body_parts[i]

            # prompt the user to click on the body part
            message = 'Click on the ' + body_part + ' of the animal'
            cv2.displayOverlay('image', message, 10000)  # Display the message for 10 seconds

            # Clear the coordinates
            coords.clear()        

            while True:
                # wait for the user to click on the image

                # press ESC too exit the program
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

                # press "r" to re-select the last point
                if k == ord('r'):
                    
                    i -= 1

                    coordinates[frames.index(frame), 2*(i)] = 0
                    coordinates[frames.index(frame), 2*(i)+1] = 0

                    # Display the image
                    img = cv2.imread(frame)
                    img = scale_image(img, scale_factor)
                    cv2.setMouseCallback('image', draw_circle, img)   
                    cv2.imshow('image', img)

                    # redraw the circles from 0 to i
                    for j in range(i):
                        x = int(coordinates[frames.index(frame), 2*j] * scale_factor)
                        y = int(coordinates[frames.index(frame), 2*j+1] * scale_factor)
                        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                        cv2.imshow('image', img)        
                    
                    break

                # Print the coordinates
                if coords:

                    # adjust the coordinates to the original size of the image
                    coords[0] = (int(coords[0][0] / scale_factor), int(coords[0][1] / scale_factor))

                    # store the coordinates in the array              
                    coordinates[frames.index(frame), 2*(i)] = coords[0][0]
                    coordinates[frames.index(frame), 2*(i)+1] = coords[0][1]
                    i += 1
                    break
            
            if i == n_body_parts:
                # save the labelled frame as a png file
                frame_name = os.path.basename(frame)
                frame_path = os.path.join(dir, 'labeled_frames', frame_name)
                cv2.imwrite(frame_path, img)
                print(frame_name, 'saved')


    # Close all windows
    cv2.destroyAllWindows()

    # convert the coordinates to a dataframe. The first column will be the frame name,
    # and the remaining columns will be the coordinates of the body parts
    df = pd.DataFrame(coordinates)

    # insert the frame names as the first column
    df.insert(0, 'frame', frame_names)

    # set the column names
    df.columns = ['frame'] + [item for sublist in [[part + '_x', part + '_y'] for part in body_parts] for item in sublist]

    # save the dataframe to a csv file
    csv_file = os.path.join(dir, 'coordinates.csv')
    df.to_csv(csv_file, index=False)

if __name__ == '__main__':

    label_frames()
    pass        
        


        