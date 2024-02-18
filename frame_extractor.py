# https://stackoverflow.com/questions/21983062/in-python-opencv-is-there-a-way-to-quickly-scroll-through-frames-of-a-video-all

import cv2
import os

def extract_frames(video_path=None):
    
    if video_path == None:
        video_path = '/media/jake/LaCie/2023-05-03/videos/trial_2023_05_03_10h11m.avi'

    # get the name of the video, without the extension
    video_name = os.path.basename(video_path).split('.')[0]

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get the parent directory of the file
    parent_dir = os.path.dirname(video_path)

    # if folder "extracted_frames" does not exist, create it
    if not os.path.exists(os.path.join(parent_dir, 'extracted_frames')):
        os.makedirs(os.path.join(parent_dir, 'extracted_frames'))

    def onChange(trackbarValue):
        cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
        err,img = cap.read()
        cv2.imshow("mywindow", img)
        pass

    cv2.namedWindow('mywindow')
    cv2.createTrackbar( 'frame', 'mywindow', 0, length, onChange )

    onChange(0)
    cv2.waitKey()

    while cap.isOpened():

        frame_num = cv2.getTrackbarPos('frame','mywindow')

        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)

        err,img = cap.read()
        if err == False:
            break
        
        cv2.imshow("mywindow", img)

        key = cv2.waitKey(1) & 0xFF

        # if the user presses the 's' key, save the frame as a .png file 
        if key == ord('s'):
            frame_name = f'{video_name}_frame_{frame_num}.png'
            frame_path = os.path.join(parent_dir, 'extracted_frames', frame_name)

            cv2.imwrite(frame_path,img)
            print(frame_name, 'saved')


        k = cv2.waitKey(10) & 0xff
        if k==27:
            break


if __name__ == '__main__':
    extract_frames()