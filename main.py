from trackers import Tracker
from utils import saveVideo, readVideo
from ultralytics import YOLO
import cv2
from teamAssigner import TeamAssigner
import numpy as np
from cameraMovement import CameraMovementEstimation
def main():
    videoFrames = readVideo('input videos/08fd33_4.mp4')
    tracker = Tracker('models/100EpochModel.pt')

    tracks = tracker.getObjectTracks(videoFrames)
    tracks['ball'] = tracker.interpolateBallPosition(tracks['ball'])

    assign_team_colors(videoFrames, tracks)
    teamBallControl = playerBallAssigner(tracker, tracks)

    outputVideoFrames = tracker.drawAnnotations(videoFrames, tracks, teamBallControl)
    outputVideoFrames=setCameraMovement(outputVideoFrames)
    

    saveVideo(outputVideoFrames, 'outputVideos/outputVideo.avi')


def assign_team_colors(videoFrames, tracks):
    teamAssigner = TeamAssigner()
    teamAssigner.assignTeamColors(videoFrames[0], tracks['players'][0])
    
    for frameNum, playerTrack in enumerate(tracks['players']):
        for playerID, track in playerTrack.items():
            team = teamAssigner.getPlayerTeams(videoFrames[frameNum], track['bounding box'], playerID)
            tracks['players'][frameNum][playerID]['team'] = team
            tracks['players'][frameNum][playerID]['teamColor'] = teamAssigner.teamColors[team]


def playerBallAssigner(tracker, tracks):
    teamBallControl = []
    for frameNum, playerTrack in enumerate(tracks['players']):
        ballBox = tracks['ball'][frameNum][1]['bounding box']
        assignedPlayer = tracker.playerBallAssigner(playerTrack, ballBox)

        if assignedPlayer != -1:
            tracks['players'][frameNum][assignedPlayer]['has ball'] = True
            teamBallControl.append(tracks['players'][frameNum][assignedPlayer]['team'])
        else:
            if teamBallControl:
                teamBallControl.append(teamBallControl[-1])
            else:
                continue
    teamBallControl = np.array(teamBallControl)
    return teamBallControl
            

def setCameraMovement(videoFrames):
    cameraMovementEstimator = CameraMovementEstimation(videoFrames[0])
    movementPerFrame = cameraMovementEstimator.getCameraMovement(videoFrames)
    outPutFrames = cameraMovementEstimator.drawCameraMovement(videoFrames, movementPerFrame)
    return outPutFrames
     


if __name__ == "__main__":
    
    main()
