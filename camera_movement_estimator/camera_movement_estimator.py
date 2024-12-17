import pickle
import cv2
import numpy as np
import sys
import os
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        # Minimum movement threshold (in pixels) to consider camera movement significant
        self.minimum_distance = 5
        
        # Lucas-Kanade Optical Flow algorithm
        self.lk_params = dict(
            winSize = (15, 15),  # Size of the search window at each pyramid level
            maxLevel = 2,        # Maximum number of pyramid levels for optical flow
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria
        )

        # Convert the first frame to grayscale, as optical flow works on single-channel images
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask to limit the areas where features are detected
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1      # Allow features only on the left edge of the frame
        mask_features[:, 900:1050] = 1  # Allow features only on the right edge of the frame

        # Parameters for detecting features in the first frame
        self.features = dict(
            maxCorners = 100,     # Maximum number of corners (features) to detect
            qualityLevel = 0.3,   # Minimum quality of corners (relative to the best corner)
            minDistance = 3,      # Minimum distance between detected features
            blockSize = 7,        # Block size for corner detection
            mask = mask_features, # Mask to limit feature detection to certain areas
        )
    
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Estimates the camera's movement for a sequence of video frames.

        Parameters:
        - frames: A list of video frames (in BGR format).
        - read_from_stub: Whether to read precomputed motion data from a stub file.
        - stub_path: Path to the stub file for saving/loading camera movement data.

        Returns:
        - camera_movement: A list of [x, y] camera movement vectors for each frame.
        """
        # Check if a stub file exists and load precomputed data if requested
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize a list to store camera movement for each frame
        camera_movement = [[0, 0]] * len(frames)  # Initially, no movement is assumed

        # Convert the first frame to grayscale for optical flow calculations
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        # Detect good features to track in the first frame using the parameters defined in __init__
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Process each frame sequentially to calculate camera movement
        for frame_num in range(1, len(frames)):
            # Convert the current frame to grayscale
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Calculate optical flow to track the movement of features between the two frames
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray,         # Previous frame (grayscale)
                frame_gray,       # Current frame (grayscale)
                old_features,     # Features to track from the previous frame
                None,             # Placeholder for the output status/error
                **self.lk_params  # Lucas-Kanade parameters
            )

            # Initialize variables to track the maximum distance and movement
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # Iterate through each pair of tracked points (new and old features)
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                # Extract the (x, y) coordinates of the feature points
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # Calculate the Euclidean distance between the points in consecutive frames
                distance = measure_distance(new_features_point, old_features_point)

                # If the current feature has the maximum distance, update the movement vector
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            # If the maximum distance exceeds the threshold, record the camera movement
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]

                # Update the features to track in the next frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            # Update the old frame to the current frame
            old_gray = frame_gray.copy()
        
        # Save the computed camera movement data to a stub file if a path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f'Camera movement X: {x_movement:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f'Camera movement Y: {y_movement:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)
        return output_frames