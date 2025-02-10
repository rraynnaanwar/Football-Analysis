from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import getCenterOfBox, getWidthOfBox, measureDistance, getFootPosition
class Tracker:
    def __init__(self, modelPath):
        self.model = YOLO(modelPath, verbose=False)
        self.tracker = sv.ByteTrack()
    
    def addPositionsToTracks(self, tracks):
        for object, objectTracks in tracks.items():
            for frameNum, track in enumerate(objectTracks):
                for trackID, trackInfo in track.items():
                    boundingBox = trackInfo['bounding box']
                    if object == 'ball':
                        position = getCenterOfBox(boundingBox)
                    else:
                        position = getFootPosition(boundingBox)
                    tracks[object][frameNum][trackID]['position'] = position
        


    def detectFrames(self, frames):
            batchSize = 20
            detections = []
            for i in range(0, len(frames), batchSize):
                temp = self.model.predict(frames[i:i+batchSize], conf=0.1, verbose =False)
                detections += temp
            return detections

    def interpolateBallPosition(self,ballPositions):
        ballPositions = [x.get(1,{}).get('bounding box', []) for x in ballPositions]
        dfBallPositions = pd.DataFrame(ballPositions, columns = ['x1', 'y1', 'x2', 'y2'])
        dfBallPositions = dfBallPositions.interpolate()
        dfBallPositions = dfBallPositions.bfill()
        ballPositions = [{1:{'bounding box' :x}}for x in dfBallPositions.to_numpy().tolist() ]
        return ballPositions
    
    def getObjectTracks(self,frames):

        detections = self.detectFrames(frames)
        
        
        tracks = {
            "players" : [],
            "referees" : [], 
            "ball" : []
        } 
        for frameNum, detection in enumerate(detections):
            className = detection.names
            classNameInv = {v:k for k,v in className.items()}
            detectionSupervision = sv.Detections.from_ultralytics(detection)

            for objectIndex, classID in enumerate(detectionSupervision.class_id):
                if className[classID] == "goalkeeper":
                    detectionSupervision.class_id[objectIndex] = classNameInv["player"]
            
            detectionWithTracks = self.tracker.update_with_detections(detectionSupervision)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            for frameDetection in detectionWithTracks:
                boundingBox = frameDetection[0].tolist()
                classID = frameDetection[3]
                trackID = frameDetection[4]
                if classID == classNameInv['player']:
                    tracks["players"][frameNum][trackID] = {"bounding box":boundingBox}

                if classID == classNameInv['referee']:
                    tracks["referees"][frameNum][trackID] = {"bounding box":boundingBox}

            for frameDetection in detectionSupervision:
                boundingBox = frameDetection[0].tolist()
                classID = frameDetection[3]
                if classID == classNameInv['ball']:
                    tracks["ball"][frameNum][1] = {"bounding box":boundingBox}
        
        return tracks
    
    def drawAnnotations(self, videoFrames, tracks, teamBallControl):
        outputVideoFrames = []
        for frameNum, frame in enumerate(videoFrames):
            frame = frame.copy()

            # Handle cases where tracking data might be missing
            if frameNum >= len(tracks['players']):
                playerDict = {}
            else:
                playerDict = tracks['players'][frameNum]

            if frameNum >= len(tracks['referees']):
                refereeDict = {}
            else:
                refereeDict = tracks['referees'][frameNum]

            if frameNum >= len(tracks['ball']):
                ballDict = {}
            else:
                ballDict = tracks['ball'][frameNum]

            for trackID, player in playerDict.items():
                color = player.get("teamColor", (0,0,255))
                frame = self.drawEllipse(frame, player["bounding box"], color, trackID)
                if player.get('has ball', False):
                    frame = self.drawTriangle(frame, player['bounding box'], (0,0,255))
                    

            for __, referee in refereeDict.items():
                frame = self.drawEllipse(frame, referee["bounding box"], (0, 255, 255), None)

            for _, ball in ballDict.items():
                frame = self.drawTriangle(frame, ball["bounding box"], (0,255,0))

            frame = self.drawTeamBallControl(frame, frameNum, teamBallControl)

            outputVideoFrames.append(frame)
        return outputVideoFrames

    def drawTeamBallControl(self, frame, frameNum, teamBallControl):

        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0,frame)
        teamBallControlTillFrame = teamBallControl[:frameNum+1]
        team1NumFrames = teamBallControlTillFrame[teamBallControlTillFrame==1].shape[0]
        team2NumFrames = teamBallControlTillFrame[teamBallControlTillFrame==2].shape[0]
        team1Posession  =team1NumFrames/(team1NumFrames+team2NumFrames)
        team2Posession = team2NumFrames/(team1NumFrames+team2NumFrames)
        cv2.putText(frame, f"Team 1 Ball Control: {team1Posession*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team2Posession*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 3)
        return frame


    def drawEllipse(self, frame, boundingBox, color, trackID):
        y2= int(boundingBox[3])
        xCenter,yCenter = getCenterOfBox(boundingBox)
        width = getWidthOfBox(boundingBox)
        cv2.ellipse(
        frame,
        center=(xCenter, y2),
        axes=(width, int(0.35 * width)),
        angle=0.0,
        startAngle=45,
        endAngle=300,
        color=color,
        thickness=1,
        lineType=cv2.LINE_4
    )


        return frame
    
    def drawTriangle(self, frame, boundingBox, color):
        y= int(boundingBox[1])
        x,_ = getCenterOfBox(boundingBox)
        trianglePoints = np.array( [[x,y], [x+10,y-20], [x-10,y-20]])
        cv2.drawContours(frame, [trianglePoints], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [trianglePoints], 0, (0,0,0),2)
        return frame
    
    def playerBallAssigner(self, players, ballBoundingBox):
        maxDistance = 70
        ballPosition = getCenterOfBox(ballBoundingBox)
        minimumDistance = 99999
        assignedPlayer = -1
        for playerID, player in players.items():
            playerBoundingBox = player['bounding box']
            distanceLeft = measureDistance((playerBoundingBox[0], playerBoundingBox[-1]),(ballPosition))
            distanceRight = measureDistance((playerBoundingBox[2], playerBoundingBox[-1]),(ballPosition))
            distance = min(distanceLeft, distanceRight)
            if distance<maxDistance:
                if distance<minimumDistance:
                    minimumDistance = distance
                    assignedPlayer = playerID
        return assignedPlayer

    