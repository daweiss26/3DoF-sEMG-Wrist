"""
This script runs wrist tracking on the Orbita3D using Mediapipe landmarking libraries.
The landmarks provide a rotation matrix for elbow-to-wrist to wrist-to-hand.
This rotation matrix is then converted to a quaternion command and sent to the Orbita3D.
"""

from orbita_controller import Orbita
from landmarker import Landmarker
from transformer import Transformer
from cv2 import CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
import time
SAFETY_RAD_LIMIT = 0.45 # Max singular roll and pitch is 0.45 (~27deg)
RAD_LOWER_LIMIT = 0.1 # 2-3deg min movement
REST_INTERVAL = 0 # Optimal is 0
THETA_ERR_LIMIT = 1e-6


def main():
    with Orbita('./config/default.yaml') as orbita:
        orbita.wake_up()
        # orbita.stretch()

        with Landmarker(1, './task/hand_landmarker.task', './task/pose_landmarker_lite.task') as landmarker:
            frame_width = int(landmarker.camera.get(CAP_PROP_FRAME_WIDTH))
            frame_height = int(landmarker.camera.get(CAP_PROP_FRAME_HEIGHT))
            transformer = Transformer(frame_width, frame_height)

            rest_time = time.time()
            while landmarker.camera.isOpened():
                landmarking_results, _, _ = landmarker.run_detection()
                if landmarking_results == None:
                    break

                rotation_matrices = []
                if time.time() - rest_time > REST_INTERVAL:
                    for landmarking_result in landmarking_results:
                        rotation_matrices.append(transformer.get_R_from_landmarks(landmarking_result))
                    rest_time = time.time()

                ### METHOD 1: RPY
                # rpys = []
                # if rotation_matrices:
                #     current_q = orbita.get_orientation()
                #     current_R = transformer.get_R_from_q(current_q)
                #     for R in rotation_matrices:
                #         theta_from_home = transformer.get_theta_from_R(R)
                #         theta_step = transformer.get_theta_from_R(R @ current_R.T)
                #         if theta_from_home < SAFETY_RAD_LIMIT and theta_step > RAD_LOWER_LIMIT:
                #             rpys.append(transformer.get_rpy(R))

                # if rpys:
                #     print('Command:', rpys)
                #     orbita.set_rpy_orientation(rpys[0])
                #     print('Orientation:', orbita.get_rpy_orientation())
                ###

                ### METHOD 2: Q
                quaternions = []
                if rotation_matrices:
                    current_q = orbita.get_orientation()
                    for R in rotation_matrices:
                        q = transformer.get_q_from_R(R)
                        tilt_from_home = transformer.get_tilt_from_R(R)
                        theta_step = transformer.get_theta_between_q(current_q, q)
                        if tilt_from_home < SAFETY_RAD_LIMIT and theta_step > RAD_LOWER_LIMIT:
                            quaternions.append(q)

                if quaternions:
                    orbita.set_orientation(quaternions[0])
                ###

        orbita.go_to_sleep()

if __name__ == "__main__":
    main()
