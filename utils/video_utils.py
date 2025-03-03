import cv2
import os
# read a video frame by frame
def read_video(path):

    # create a videoCapture object
    cap = cv2.VideoCapture(path)

    print("Video file exists:", os.path.exists(path))


    # Loop until the end of the video
    frames = []
    while (cap.isOpened()):
        # array to store the frames captured
       
        # cap.read() returns two values: a boolean value saved in ret which is true if frame was successfully captured
        # and frame is the actual image captured from the video
        ret , frame = cap.read()
       
        if not ret:
            break
        frames.append(frame)

        

    return frames

# parameters: an array of frames and a path to the output video
def save_video(output_video_frames, output_video_path):
   

    # specify the video compression format as 'XVID' 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(
        output_video_path, # video is saved in this path
        fourcc, # video compression format
        24, # number of frames per second
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0] ) ) # Frame size is set to the height and width of the first frame
    
    for frame in output_video_frames:
        # write each frame to the videowriter
        output.write(frame)

    output.release()


