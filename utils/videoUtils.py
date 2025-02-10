import cv2
#this function takes in the path of a video and returns a list of the frames
def readVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    frames = []
    while True:
        ret, frame =cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
#this function creates a video file given the frames
def saveVideo(outputVideoFrames, outputVideoPath):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #parameters, path, video format, fps, width, height for each frame
    out = cv2.VideoWriter(outputVideoPath, fourcc, 24, (outputVideoFrames[0].shape[1], outputVideoFrames[0].shape[0]))
    for frame  in outputVideoFrames:
        out.write(frame)
    out.release()