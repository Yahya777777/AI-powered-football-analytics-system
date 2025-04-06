import cv2
import numpy as np
from utils import measure_distance, measure_xy_distance
class CameraMovementEstimator():

    def __init__(self,frame):
        # let's ignore every movement less that 5 pixels
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Create a black image (all 0s) the same size as the frame. This will be used as a mask to tell OpenCV where to look for things to track.
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )

    

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    # find the camera movement in specific frame
                    camera_movement = camera_movement_per_frame[frame_num]
                    # get the adjusted position according to the movement
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames):

        # create a list the same size as the number of frames and initialize all to zero (at first camera didnt move)
        camera_movement = [[0,0]]*len(frames)

        # we will take the first frame and turn it into black and white(it is easier to track things in black and white)
        first_frame = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        # find corners in the grey image that we can track later
        old_features = cv2.goodFeaturesToTrack(first_frame, **self.features)

        # Loop through the rest of the frames:
        for frame_num in range(1, len(frames)):

            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            # Get the new position of each corner dot to  check if each of the corners moved in new frames.
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(first_frame, frame_gray, old_features, None, **self.lk_params)

            # To remember the biggest movement we see:
            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            # loop through the old and new features:
            for i, (old, new) in enumerate(zip(new_features, old_features)):
                # for every pair of old and new corners, we turn the points into normal X, Y coordinates we can use easily.
                old_corner_point = old.ravel()
                new_corner_point = new.ravel()

                # measure how far the dot has moved:
                distance = measure_distance(new_corner_point, old_corner_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_corner_point, new_corner_point)
            #If the camera really did move a noticeable amount, we store that camera movement for this frame!
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                #  Now we pick new dots to track from this new frame, because we want fresh clues for the next round.
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
            #The new frame becomes our new “previous” frame to compare with the next one.
            old_gray = frame_gray.copy()

        return  camera_movement
    
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames
        






