import cv2
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        # we will calculate the speed and distance evry 5 frames and with rate 24/s
        self.frame_window=5
        self.frame_rate=24
    
    def add_speed_and_distance_to_tracks(self,tracks):
        #Initializes a dictionary to keep track of the total distance covered by each object
        total_distance= {}

        for object, object_tracks in tracks.items():
            # We only want speed and distance for players
            if object == "ball" or object == "referees":
                continue 

            number_of_frames = len(object_tracks)
            # loop over batches of windows(5)
            for frame_num in range(0,number_of_frames, self.frame_window):
                # Calculates the last frame of the current batch while ensuring it doesn’t go out of bounds.
                # Example: if frame_num = 48, frame_num + 5 = 53, but you only have 50 frames → clip it to 49.
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    # to calculate the speed and distance player has to exist in the first and last frame of the batch
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']
                    # if its not inside the trapazoid shape, dont measure it
                    if start_position is None or end_position is None:
                        continue
                    #Calculates the Euclidean distance between the two positions (likely in meters).
                    #This is the distance traveled by that player during this frame window. 
                    distance_covered = measure_distance(start_position,end_position)
                    # Calculates the time taken between those two frames in seconds, using the frame rate of the video 
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    # Computes the speed of the player in meters per second, then converts to km/h.
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6
                    
                    #Initializes the dictionary for this object type if it doesn't already exist.
                    if object not in total_distance:
                        total_distance[object]= {}
                    #Initializes the cumulative distance for this player ID.
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    #Updates the total distance covered by this player.
                    total_distance[object][track_id] += distance_covered
                    
                    #Loops through every frame in the current frame window.
                    for frame_num_batch in range(frame_num,last_frame):
                        # If the player doesn’t appear in one of the in-between frames, skip adding speed/distance to that frame.
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        #Adds the calculated speed and total distance covered so far to the tracking data for that player in each frame of the batch.
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees" or object == "goalkeeper":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       #Converts the position to a list (so we can modify it).
                       position = list(position)
                       #Moves the label slightly below the feet to avoid overlapping the player (adds 40 pixels to the y-coordinate).
                       position[1]+=40
                        #Converts the modified position back to a tuple of integers (OpenCV needs integer coords).
                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            output_frames.append(frame)
        
        return output_frames