# https://stackoverflow.com/questions/21983062/in-python-opencv-is-there-a-way-to-quickly-scroll-through-frames-of-a-video-all

import cv2
import os
import time

from pynput import keyboard

def extract_frames(video_dir):

    # This will hold the currently pressed keys
    current_keys = set()

    def on_press(key):
        current_keys.add(str(key))

    def on_release(key):
        current_keys.remove(str(key))

    # Start the keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    
    # get the list of video files in the directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

    n_videos = len(video_files)
    if n_videos == 0:
        print('No video files found in the directory')
        return
    
    print(f'Found {n_videos} video files in the directory')

    # if folder "extracted_frames" does not exist, create it
    if not os.path.exists(os.path.join(video_dir, 'extracted_frames')):
        os.makedirs(os.path.join(video_dir, 'extracted_frames'))


    # loop through the video files
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        
        # get the name of the video, without the .avi extension
        video_name = os.path.basename(video_path).split('.avi')[0]

        print(f'Extracting frames from {video_name}...')

        # open the video file
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print the frame size
        print('Frame size:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
      
        def onChange(trackbarValue):
            if trackbarValue > length:
                trackbarValue = length
                
            cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
            err,img = cap.read()
            if err:
                cv2.imshow("mywindow", img)
            else:
                print('Reached the end of the video')
                # cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue-2)

        cv2.namedWindow('mywindow')
        cv2.createTrackbar( 'frame', 'mywindow', 0, length, onChange )

        onChange(0)
        # cv2.waitKey()


        while cap.isOpened():

            frame_num = cv2.getTrackbarPos('frame','mywindow')
            if frame_num >= length:
                frame_num -= 1

            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)

            err,img = cap.read()
            if err == False:
                break
            
            cv2.imshow("mywindow", img)
            cv2.waitKey(1)  # Add this line

            # key = cv2.waitKey(1) & 0xFF

            # if the user presses the 's' key, save the frame as a .png file 
            # if key == ord('s'):
            if "'s'" in current_keys:
                frame_name = f'{video_name}_frame_{frame_num}.png'
                frame_path = os.path.join(video_dir, 'extracted_frames', frame_name)

                cv2.imwrite(frame_path,img)
                print(frame_name, 'saved')
                time.sleep(0.2)  # Add this line

            
            # if the user presses the 'n' key, close the video and proceed to 
            # the next video in the loop
            # if key == ord('n'):
            if "'n'" in current_keys:
                # close the video
                print('Closing video...')
                cap.release()
                cv2.destroyAllWindows()
                time.sleep(0.2)  # Add this line
                continue


            # if the user presses the 'esc' key, exit the loop
            # k = cv2.waitKey(10) & 0xff
            if keyboard.Key.esc in current_keys:
            # if k==27:
                break


if __name__ == '__main__':
    video_dir = '/media/jake/LaCie/video_files'
    extract_frames(video_dir)