"""Uses Mediapipe to extract hand and elbow landmarks from livestreamed images"""

import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles, HandLandmarksConnections
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark


class Landmarker:
    DEFAULT_CONNECTIONS = list(vision.HandLandmarksConnections.HAND_CONNECTIONS) + [HandLandmarksConnections.Connection(start=0,end=21)]
    DEFAULT_HAND_CONNECTIONS = list(vision.HandLandmarksConnections.HAND_CONNECTIONS) + [(0,21)]
    DEFAULT_CONNECTION_STYLE = dict(drawing_styles.get_default_hand_connections_style())
    DEFAULT_CONNECTION_STYLE[(0, 21)] = drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
    DEFAULT_LANDMARK_STYLE = dict(drawing_styles.get_default_hand_landmarks_style())
    DEFAULT_LANDMARK_STYLE[0] = drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3)
    DEFAULT_LANDMARK_STYLE[1] = drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3)
    DEFAULT_LANDMARK_STYLE[5] = drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3)
    DEFAULT_LANDMARK_STYLE[9] = drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3)
    DEFAULT_LANDMARK_STYLE[13] = drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3)
    DEFAULT_LANDMARK_STYLE[17] = drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3)
    DEFAULT_LANDMARK_STYLE[21] = drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3)
    ELBOW_VISIBILITY_THRESHOLD = 0.8
    hand_landmarker = None
    hand_result = None
    pose_landmarker = None
    pose_result = None

    def __init__(self, camera_index, hand_task, pose_task, use_elbow=False):
        self.camera = cv2.VideoCapture(camera_index)
        hand_options = vision.HandLandmarkerOptions(
                            base_options=python.BaseOptions(model_asset_path=hand_task),
                            num_hands=1,
                            min_hand_detection_confidence=0.8,
                            min_hand_presence_confidence=0.8,
                            min_tracking_confidence=0.8,
                            running_mode=vision.RunningMode.LIVE_STREAM,
                            result_callback=self.set_hand_result)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        if use_elbow:
            pose_options = vision.PoseLandmarkerOptions(
                                base_options=python.BaseOptions(model_asset_path=pose_task),
                                num_poses=1,
                                min_pose_detection_confidence=0.8,
                                min_pose_presence_confidence=0.8,
                                min_tracking_confidence=0.8,
                                running_mode=vision.RunningMode.LIVE_STREAM,
                                result_callback=self.set_pose_result)
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            self.DEFAULT_CONNECTIONS.append((0,22))
            self.DEFAULT_CONNECTION_STYLE[(0, 22)] = drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
            self.DEFAULT_LANDMARK_STYLE[22] = drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.camera.release()
        cv2.destroyAllWindows()

    def get_hand_result(self):
        return self.hand_result

    def set_hand_result(self, result, output_image: mp.Image, timestamp_ms: int):
        self.hand_result = (result, output_image.numpy_view(), timestamp_ms)

    def get_pose_result(self):
        return self.pose_result

    def set_pose_result(self, result, output_image: mp.Image, timestamp_ms: int):
        self.pose_result = (result, output_image.numpy_view(), timestamp_ms)

    def get_landmarks(self, output_image, hand_result, pose_result, visualize, 
                      connections, connection_style, landmark_style):
        """Extracts landmarks from the detection results and optionally draws them"""
        landmarking_results = []
        color_flipped = False
        left_found = False
        right_found = False
        for hand_idx in range(len(hand_result.hand_landmarks)):
            landmarks = []
            handedness = hand_result.handedness[hand_idx][0].category_name
            elbow = None
            if (handedness == 'Left' and left_found) or (handedness == 'Right' and right_found):
                continue
            elif pose_result and handedness == 'Left' and not left_found:
                elbow = pose_result.pose_landmarks[0][13]
                left_found = True
            elif pose_result and handedness == 'Right' and not right_found:
                elbow = pose_result.pose_landmarks[0][14]
                right_found = True

            if elbow and elbow.visibility > self.ELBOW_VISIBILITY_THRESHOLD:
                landmarks = list(hand_result.hand_landmarks[hand_idx])
                landmarks.append(NormalizedLandmark(
                    x=(3*hand_result.hand_landmarks[hand_idx][9].x+2*hand_result.hand_landmarks[hand_idx][13].x)/5,
                    y=(3*hand_result.hand_landmarks[hand_idx][9].y+2*hand_result.hand_landmarks[hand_idx][13].y)/5,
                    z=(3*hand_result.hand_landmarks[hand_idx][9].z+2*hand_result.hand_landmarks[hand_idx][13].z)/5,
                    visibility=1.0
                )) # End of wrist orientation
                landmarks.append(elbow)
                landmarking_results.append((landmarks, handedness))
            elif not elbow:
                landmarks = list(hand_result.hand_landmarks[hand_idx])
                landmarks.append(NormalizedLandmark(
                    x=(3*hand_result.hand_landmarks[hand_idx][9].x+2*hand_result.hand_landmarks[hand_idx][13].x)/5,
                    y=(3*hand_result.hand_landmarks[hand_idx][9].y+2*hand_result.hand_landmarks[hand_idx][13].y)/5,
                    z=(3*hand_result.hand_landmarks[hand_idx][9].z+2*hand_result.hand_landmarks[hand_idx][13].z)/5,
                    visibility=1.0
                )) # End of wrist orientation
                landmarking_results.append((landmarks, handedness))


            if visualize and landmarks:
                output_image = self.visualize_results(output_image, landmarks, connections, connection_style, landmark_style)
                color_flipped = True
        
        return landmarking_results, output_image, color_flipped


    def run_detection(self, visualize=True, connections=None, connection_style=None, landmark_style=None):
        """Detects hand and pose landmarks in the image captured by the camera"""
        landmarking_results = []
        timestamp = time.time()
        success, bgr_image = self.camera.read()
        if not success:
            return []
        bgr_image.flags.writeable = False
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.hand_landmarker.detect_async(mp_image, int(time.time()*1000))
        if self.pose_landmarker: self.pose_landmarker.detect_async(mp_image, int(time.time()*1000))

        if self.hand_result is not None and (self.pose_result is not None or not self.pose_landmarker):
            hand_result, output_image, timestamp = self.hand_result
            if self.pose_result: pose_result, _, _  = self.pose_result
            else: pose_result = None

            color_flipped = False
            if hand_result.hand_landmarks and (pose_result or not self.pose_landmarker):
                landmarking_results, output_image, color_flipped = self.get_landmarks(output_image, hand_result, pose_result, visualize, 
                                                                       connections, connection_style, landmark_style)

            if visualize:
                if not color_flipped:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("IT'S ORBITA TIME!!!", output_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None, None, False
        elif key == ord('s'):
            return None, None, True
        
        return landmarking_results, timestamp, None
    

    def visualize_results(self, image, landmarks, connections, connection_style, landmark_style):
        """Draws the detected landmarks and connections on the provided image"""
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.ascontiguousarray(image.copy())
        drawing_utils.draw_landmarks(
            image,
            landmarks,
            connections=connections if connections else self.DEFAULT_CONNECTIONS,
            connection_drawing_spec=connection_style if connection_style else self.DEFAULT_CONNECTION_STYLE,
            landmark_drawing_spec=landmark_style if landmark_style else self.DEFAULT_LANDMARK_STYLE,
        )
        return image


    def run_hand_detection(self, visualize=True, connections=None, connection_style=None, landmark_style=None):
        """Detects hand landmarks in the image captured by the camera"""
        landmarking_results = []
        timestamp = None
        success, bgr_image = self.camera.read()
        if not success:
            return []
        bgr_image.flags.writeable = False
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.hand_landmarker.detect_async(mp_image, int(time.time()*1000))

        if self.hand_result is not None:
            hand_result, output_image, timestamp = self.hand_result

            if hand_result.hand_landmarks:
                landmarking_results, output_image = self.get_hand_landmarks(output_image, hand_result, visualize, 
                                                                            connections, connection_style, landmark_style)

            if visualize:
                cv2.imshow("IT'S ORBITA TIME!!!", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

        return landmarking_results, timestamp
    

    def get_hand_landmarks(self, output_image, hand_result, visualize, 
                           connections, connection_style, landmark_style):
        """Extracts hand landmarks from the detection results and optionally draws them"""
        landmarking_results = []
        left_found = False
        right_found = False
        for hand_idx in range(len(hand_result.hand_landmarks)):
            landmarks = []
            handedness = hand_result.handedness[hand_idx][0].category_name
            if (handedness == 'Left' and left_found) or (handedness == 'Right' and right_found):
                continue
            elif handedness == 'Left' and not left_found:
                left_found = True
            elif handedness == 'Right' and not right_found:
                right_found = True

            landmarks = list(hand_result.hand_landmarks[hand_idx])
            landmarking_results.append((landmarks, handedness))

            if visualize and landmarks:
                output_image = self.visualize_hand_results(output_image, landmarks, connections, connection_style, landmark_style)

        return landmarking_results, output_image

    
    def visualize_hand_results(self, image, landmarks, connections, connection_style, landmark_style):
        """Draws the detected hand landmarks and connections on the provided image"""
        image = np.ascontiguousarray(image.copy())
        drawing_utils.draw_landmarks(
            image,
            landmarks,
            connections=connections if connections else self.DEFAULT_HAND_CONNECTIONS,
            connection_drawing_spec=connection_style if connection_style else self.DEFAULT_CONNECTION_STYLE,
            landmark_drawing_spec=landmark_style if landmark_style else self.DEFAULT_LANDMARK_STYLE,
        )
        return image
