from utils import save_video, read_video
from trackers import Tracker
def main():
    # read frames
    frames = read_video('input/match_clip.mp4')
    #create a tracker instance
    tracker = Tracker("models/best.pt")
    
    t = tracker.get_object_track(frames, True, "tracker.pk1")

    output = tracker.draw_new_boundingBox(frames, t)

    save_video(output, 'output/output.avi')

if __name__ == '__main__':
    main()