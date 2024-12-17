import cv2

def read_video(video_path):
    '''
    Takes in a video path and return a list of frames for the video
    '''
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read() # reads in next frame
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter.fourcc(*'XVID') # defining output format
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) # frames, xy pos
    for frame in output_video_frames:
        out.write(frame)
    out.release()