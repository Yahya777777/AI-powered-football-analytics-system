from utils import save_video, read_video
from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator
from team_assigner import TeamAssigner
from speed_distance import SpeedAndDistance_Estimator
from transform_view import ViewTransformer
def main():
    # read frames
    video_frames = read_video('input/Recording 2025-04-05 020634.mp4')
    #create a tracker instance
    tracker = Tracker("models/best.pt")
    
    t = tracker.get_object_track(video_frames, True, None)

    tracker.add_position_to_tracks(t)
    # estimate camera moveemnt
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
                                            
    camera_movement_estimator.add_adjust_positions_to_tracks(t,camera_movement_per_frame)
     # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(t)

     # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(t)
    # assign teams to players
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], t['players'][0])

    for frame_num, player_track in enumerate(t['players']):
        for player_id, track in player_track.items():
            team = team_assigner.assign_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            t['players'][frame_num][player_id]['team'] = team 
            t['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team] 

    # predict non-detected ball positions
    t["ball"] = tracker.predict_ball_position(t["ball"])
    

    output = tracker.draw_new_boundingBox(video_frames, t)
    ## Draw Camera movement
    output = camera_movement_estimator.draw_camera_movement(output,camera_movement_per_frame)

     ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output,t)


    save_video(output, 'output/output4.avi')

if __name__ == '__main__':
    main()