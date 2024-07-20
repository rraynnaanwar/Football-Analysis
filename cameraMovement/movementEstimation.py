
import cv2
import numpy as np
import sys
sys.path('../')
from utils import measureDistance, measureXYDistance
class CameraMovementEstimation:
    def __init__(self, frame):
        self.minimumDistance = 5
        firstFrameGrayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        maskFeatures = np.zeros_like(firstFrameGrayScale)
        maskFeatures[:, 0:20] = 1
        maskFeatures[:, 900:1050] = 1
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7, 
            mask = maskFeatures
        )
        self.lkParams = dict(
            windowSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03)
        )

    def getCameraMovement(self, frames):
        cameraMovement = [[0,0]*len(frames)]
        oldGray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        oldFeatures = cv2.goodFeaturesToTrack(oldGray, **self.features)
        
        for frameNum in range(1,len(frames)):
            frameGray = cv2.cvtColor(frames[frameNum], cv2.COLOR_BGR2GRAY)
            newFeatures, _,_ = cv2.calcOpticalFlowPyrLK(oldGray, frameGray, oldFeatures, None, **self.lkParams)
            maxDistance = 0
            cameraMovement_X, cameraMovement_Y =0,0
            for i, (new,old) in enumerate(newFeatures, oldFeatures):
                newFeaturesPoint = new.ravel()
                oldFeaturesPoint = old.ravel()
                distance = measureDistance(newFeaturesPoint, oldFeaturesPoint)
                if distance > maxDistance:
                    maxDistance = distance
                    cameraMovement_X, cameraMovement_Y = measureXYDistance(oldFeaturesPoint, newFeaturesPoint)
            if maxDistance>self.minimumDistance:
                cameraMovement[frameNum] = [cameraMovement_X, cameraMovement_Y]
                oldFeatures = cv2.goodFeaturesToTrack(frameGray, **self.features)
            
            oldGray = frameGray.copy()
        return cameraMovement
    



