from orbita_controller import Orbita
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# orbita = Orbita('./config.yaml')
# orbita.wake_up()
# orbita.stretch()
# orbita.dance()
# orbita.go_to_sleep()

hand_model_path = './hand_landmarker.task'
last_hand_result = None
pose_model_path = './pose_landmarker_lite.task'
last_pose_result = None

def print_pose_result(result, output_image: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result))
    global last_pose_result 
    last_pose_result = (result, output_image)

def print_hand_result(result, output_image: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result))
    global last_hand_result 
    last_hand_result = (result, output_image)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model_path),
    num_hands=2,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_hand_result)

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=pose_model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_segmentation_masks=True,
    result_callback=print_pose_result)

with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker: 
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        cap = cv2.VideoCapture(0)
        time.sleep(1)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            hand_landmarker.detect_async(mp_image, int(time.time()*1000))
            pose_landmarker.detect_async(mp_image, int(time.time()*1000))

            # Optional: Draw landmarks using MediaPipe drawing utils
            if last_pose_result is not None and last_hand_result is not None:
                pose_result, pose_output_image = last_pose_result
                hand_result, hand_output_image = last_hand_result
                np_image = pose_output_image.numpy_view()
                np_image = np.ascontiguousarray(np_image.copy())

                if hand_result.hand_landmarks and pose_result.pose_landmarks:
                    for hand_idx in range(len(hand_result.hand_landmarks)):

                        extended_landmarks = list(hand_result.hand_landmarks[hand_idx])
                        if hand_result.handedness[hand_idx][0].category_name == "Left":
                            extended_landmarks.append(pose_result.pose_landmarks[0][13])
                        else:
                            extended_landmarks.append(pose_result.pose_landmarks[0][14])

                        hand_landmark_list = landmark_pb2.NormalizedLandmarkList()
                        hand_landmark_list.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in extended_landmarks
                        ])
                        
                        custom_connections = list(mp.solutions.hands.HAND_CONNECTIONS)
                        custom_connections.append((0, 9))
                        custom_connections.append((0, 21))

                        default_conn_style = mp.solutions.drawing_styles.get_default_hand_connections_style()
                        conn_style_dict = dict(default_conn_style)
                        conn_style_dict[(0, 9)] = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
                        conn_style_dict[(0, 21)] = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)

                        default_landmark_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
                        landmark_style_dict = dict(default_landmark_style)
                        landmark_style_dict[0] = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3)
                        landmark_style_dict[1] = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3)
                        landmark_style_dict[5] = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3)
                        landmark_style_dict[9] = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3)
                        landmark_style_dict[13] = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3)
                        landmark_style_dict[17] = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3)
                        landmark_style_dict[21] = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3)

                        mp.solutions.drawing_utils.draw_landmarks(
                            np_image,
                            hand_landmark_list,
                            connections=custom_connections,
                            landmark_drawing_spec=landmark_style_dict,
                            connection_drawing_spec=conn_style_dict
                        )

                cv2.imshow('ORBITA TIME', np_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
