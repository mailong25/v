from __future__ import unicode_literals
import cv2
NUM_VIDEO_LOOP = 200

class TurnSwitching:
    def __init__(self):
        self.status = False
    def switch_on(self):
        self.status = True
    def switch_off(self):
        self.status = False
    def is_on(self):
        return self.status == True

class VideoFrame:
    def __init__(self,path_to_video):
        all_frames = []
        cap = cv2.VideoCapture(path_to_video)
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            frame = cv2.imencode('.jpg',frame)[1].tostring()
            all_frames.append(frame)
        
        self.frames = all_frames
        for i in range(0,NUM_VIDEO_LOOP):
            self.frames = self.frames + all_frames
        
    def set_frames(self,frames):
        del self.frames
        self.frames = frames
    def get_frames(self):
        return self.frames
    
switcher = TurnSwitching()
video_frames = VideoFrame('/home/ubuntu/resources/Wav2Lip/result_1_720.avi')
