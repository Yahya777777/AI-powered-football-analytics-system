from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
from utils import get_bbox_center, get_bbox_width
import numpy as np
import pandas as pd

# If
class Tracker:
    # load in the tracker and the model
    def __init__(self, model_path):
        self.model = YOLO(model_path, verbose=False)
        self.tracker = sv.ByteTrack()
       
    def predict_ball_position(self, ball_positions):
        ball_positions = [position.get(1,{}).get('bbox',[]) for position in ball_positions]
        df_ball_positiions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2']) #convert ball_positions into pandas dataframe
        df_ball_positions = df_ball_positiions.interpolate() #predictiong the in-between ball bbox
        df_ball_positions = df_ball_positiions.bfill() #replicating the nearest second detection of bbox (if the first one is missing)

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()] #extract df_ball_positiions dataframe into ball_positions list
        return ball_positions

    # detect the frames with YOLO
    # detections is an array of arrays with each array being the Coordinates of the bounding box , class id and confidence_score
    # example : [x, y , x, y , id, conf]
    def detect_frames(self, frames):
       
        # add frames in batches of 25
        # note: I first tried doing it all at once and it was way slow and time consuming, so in order to increase the speed I am doing it in batches of 25 
        batch_size = 25
        detections = []

        for i in range(0, len(frames), batch_size):
            batch_detections = self.model.predict(frames[i:i+batch_size], conf=0.1)

            detections.extend(batch_detections)
         
        
        return detections

    def draw_ellipse(self, frame, bbox, colour, track_id):

        x1, y1, x2, y2 = bbox
        x_center = int((x1+x2)/2)
        width = int(x2 - x1)

        cv2.ellipse(
            frame,
            center=(x_center, int(y2)),
            axes=(int(width), int(0.30*width)),
            angle=0.0,
            startAngle=-40,
            endAngle=240,
            color=colour,
            thickness=3,
            lineType=cv2.LINE_4,

        )
        rect_height = 25
        rect_width = 50
        # we want to make the rectangle in the center
        x1_rect = x_center - rect_width //2
        x2_rect = x_center + rect_width // 2
        y1_rect = (y2 - rect_height // 2) + 15
        y2_rect = (y2 + rect_height // 2) + 15

        if track_id != None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),  # top-left corner
                (int(x2_rect), int(y2_rect)),  # bottom-right corner
                colour,                        # color of the rectangle
                cv2.FILLED)  # inside is filled with the colour

        
            cv2.putText(
                frame,
                text= f"{track_id}", # string value to put 
                org= (int(x1_rect+15), int(y1_rect+15)), # text is inside the rectangle
                fontFace= cv2.FONT_HERSHEY_PLAIN,
                fontScale= 1,
                color=(225, 225, 225),  # colour white
                thickness= 2,
                lineType=cv2.LINE_4,

                )

        return frame

    def draw_triangle(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        # check for NaN bbox values
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            print("Invalid bounding box coordinates: NaN detected.")
            return frame  
        
        x_center, y_center = get_bbox_center(bbox)

        center = (int(x_center), int(y1))
        top_left = (int(x_center - 10), int(y1 -20))
        top_right = (int(x_center + 10), int(y1 - 20))
        
        triangle_points = np.array(
            [center, top_right, top_left]
        )
        # fill the inside of triangle with green
        cv2.drawContours(
            frame,
           [triangle_points],
            0,
            (0,255,0),
             cv2.FILLED
        )
        # outline of triangle is black
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)
       
       
        return frame
    

    # convert the detections to supervision detection format so that it can be used by ByteTrack later
    # The sv.Detections class enables easy data manipulation and filtering, and provides a consistent API for Supervision's tools like trackers, annotators, and zones.
    def get_object_track(self, frames, saved_in_file = False, pickle_path = None):

        if saved_in_file and pickle_path != None:
            with open(pickle_path, "rb") as file:
                trackers = pickle.load(file)
                
            return trackers
      
        detections = self.detect_frames(frames)

        # change the order like this: {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}
        class_name_inv = {}

        for k, v in detections[0].names.items():
            class_name_inv[v] = k
    
        print(class_name_inv)

        # create an object that will store the track id of each entity in each frame 
        trackers = {
            "players": [],
            "referees": [],
            "goalkeeper": [],
            "ball": []
        }

        
        #convert to supervision format
        for frame_id , detection in enumerate(detections):
            
           
            supervision_detection = sv.Detections.from_ultralytics(detection)

            # track objects
            detection_tracks = self.tracker.update_with_detections(supervision_detection)
            print(detection_tracks)

            trackers["players"].append({})
            trackers["goalkeeper"].append({})
            trackers["referees"].append({})
            trackers["ball"].append({})

            # for players:
            for fd in detection_tracks:
                bbox = fd[0].tolist() # get the bounding box as a python list 
                cls_id = int(fd[3]) # get the class id as int because it is a numpy number
                track_id = int(fd[4])

                if cls_id == class_name_inv["player"]:
                    trackers['players'][frame_id][track_id] = {"bbox": bbox}
                if cls_id == class_name_inv["referee"]:
                    trackers['referees'][frame_id][track_id] = {"bbox": bbox}
                if cls_id == class_name_inv["goalkeeper"]:
                    trackers['goalkeeper'][frame_id][track_id] = {"bbox": bbox}
            # we don't need to track the ball since there is only one ball
            # at first tracked the ball with other entities but it wasn't being tracked accurately and was missing from some frames ro I decided to not track it
            for sd in supervision_detection:
                bbox = sd[0].tolist()
                cls_id = int(sd[3])

                if cls_id == class_name_inv['ball']:
                    trackers['ball'][frame_id][1] = {"bbox": bbox}

            if pickle_path != None:

                with open(pickle_path, "wb") as file:
                    pickle.dump(trackers, file)
                

                

        return trackers
            
                
    def draw_new_boundingBox(self, frames, trackers):
        output_frames = []
        for frame_num, frame in enumerate(frames):

            players_dict = trackers["players"][frame_num]
            referees_dict = trackers["referees"][frame_num]
            goalkeeper_dict = trackers["goalkeeper"][frame_num]
            ball_dict = trackers["ball"][frame_num]

            for track_id, player in players_dict.items():
                draw = self.draw_ellipse(frame, player['bbox'], (0,0,0), track_id)
            

            for track_id, referee in referees_dict.items():
                draw =  self.draw_ellipse(frame, referee['bbox'], (0,0,225), None)
            
            for track_id, keeper in goalkeeper_dict.items():
                draw =  self.draw_ellipse(frame, keeper['bbox'], (0,225,0), track_id)
           
            for i, ball in ball_dict.items():
                draw = self.draw_triangle(frame, ball['bbox'])
            output_frames.append(draw)

                
                    
        return output_frames







            
         

            
       

        
