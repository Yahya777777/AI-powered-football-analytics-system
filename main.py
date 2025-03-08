from utils import save_video, read_video
from trackers import Tracker
from team_assigner import TeamAssigner
def main():
    # read frames
    video_frames = read_video('input/rca_vs_mas_test.mp4')
    #create a tracker instance
    tracker = Tracker("models/best.pt")
    
    t = tracker.get_object_track(video_frames, True, "tracker2.pk1")

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

    save_video(output, 'output/output2.avi')

if __name__ == '__main__':
    main()