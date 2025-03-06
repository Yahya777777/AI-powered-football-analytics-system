from ultralytics import YOLO
import supervision as sv

class Tracker:
    # load in the tracker and the model
    def __init__(self, model_path):
        self.model = YOLO(model_path, verbose=False)
        self.tracker = sv.ByteTrack()

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
            break
        
        return detections

    
    # convert the detections to supervision detection format so that it can be used by ByteTrack later
    # The sv.Detections class enables easy data manipulation and filtering, and provides a consistent API for Supervision's tools like trackers, annotators, and zones.
    def get_object_track(self, frames):
      
        detections = self.detect_frames(frames)

        # change the order like this: {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}
        class_name_inv = {}

        for k, v in detections[0].names.items():
            class_name_inv[v] = k
    
        print(class_name_inv)

        # we will have 
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
                bbox = fd[0].tolist()
                cls_id = int(fd[3])
                track_id = int(fd[4])

                if cls_id == class_name_inv["player"]:
                    trackers['players'][frame_id][track_id] = {"bbox": bbox}
                if cls_id == class_name_inv["referee"]:
                    trackers['referees'][frame_id][track_id] = {"bbox": bbox}
                if cls_id == class_name_inv["goalkeeper"]:
                    trackers['goalkeeper'][frame_id][track_id] = {"bbox": bbox}

            for sd in supervision_detection:
                bbox = sd[0].tolist()
                cls_id = int(sd[3])

                if cls_id == class_name_inv['ball']:
                    trackers['ball'][frame_id][1] = {"bbox": bbox}

        print(trackers["ball"])
            
                




            
         

            
       

        
