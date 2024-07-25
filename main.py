from trackers import Tracker
from utils import saveVideo, readVideo
from ultralytics import YOLO
import cv2
from teamAssigner import TeamAssigner
import numpy as np
from cameraMovement import CameraMovementEstimation


def main():
    # Read Video
    video_frames = readVideo('input videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/100EpochModel.pt')

    tracks = tracker.getObjectTracks(video_frames,)
    # Get object positions 
    tracker.addPositionsToTracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimation(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.getCameraMovement(video_frames)
    camera_movement_estimator.adJustPositions(tracks,camera_movement_per_frame)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolateBallPosition(tracks["ball"])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assignTeamColors(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.getPlayerTeams(video_frames[frame_num],   
                                                 track['bounding box'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.teamColors[team]

    
    # Assign Ball Aquisition
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bounding box']
        assigned_player = tracker.playerBallAssigner(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.drawAnnotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.drawCameraMovement(output_video_frames,camera_movement_per_frame)

   
    # Save video
    saveVideo(output_video_frames, 'outputVideos/outputVideo.avi')

if __name__ == '__main__':
    main()