"""
This script runs wrist tracking on the Orbita3D using Mediapipe landmarking libraries.
The landmarks provide a rotation matrix for elbow-to-wrist to wrist-to-hand.
This rotation matrix is then converted to a quaternion command and sent to the Orbita3D.
"""

from orbita_controller import Orbita
from landmarker import Landmarker
from transformer import Transformer
import cv2
import time
SAFETY_RAD_LIMIT = 0.45 # Max singular roll and pitch is 0.45 (~27deg)
RAD_LOWER_LIMIT = 0.05 # 2-3deg min movement
REST_INTERVAL = 0 # Optimal is 0
THETA_ERR_LIMIT = 1e-6


def main():
    with Orbita('./config.yaml') as orbita:
        orbita.wake_up()
        # time.sleep(1)
        # orbita.set_rpy_orientation((0.1, 0.1, 0.1))
        # time.sleep(1)
        # orbita.set_rpy_orientation((-0.1, -0.1, -0.1))
        # time.sleep(1)
        orbita.get_orientation()
        while True:
            print(orbita.get_rpy_orientation())

        # with Landmarker(1, './hand_landmarker.task', './pose_landmarker_lite.task') as landmarker:
        #     frame_width = int(landmarker.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     frame_height = int(landmarker.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     transformer = Transformer(frame_width, frame_height)

        #     R = transformer.rpy_to_rotation_matrix(-.4, 0, .3)
        #     print(R)
        #     q = transformer.get_q_from_R(R)
        #     print(q)
        #     R2 = transformer.get_R_from_q(q)
        #     print(R2)
        #     rpy = transformer.get_rpy_from_R(R2)
        #     print(rpy)

        #     rest_time = time.time()
        #     while landmarker.camera.isOpened():
        #         landmarking_results, _, _ = landmarker.run_detection()
        #         if landmarking_results == None:
        #             break

        #         rotation_matrices = []
        #         if time.time() - rest_time > REST_INTERVAL:
        #             for landmarking_result in landmarking_results:
        #                 rotation_matrices.append(transformer.get_R_from_landmarks(landmarking_result))
        #             rest_time = time.time()

                ### METHOD 1: RPY
                # rpys = []
                # if rotation_matrices:
                #     for R in rotation_matrices:
                #         theta = transformer.get_theta_from_R(R)
                #         if theta < SAFETY_RAD_LIMIT and theta > RAD_LOWER_LIMIT:
                #             rpys.append(transformer.get_rpy(R))

                # if rpys:
                #     print('Command:', rpys)
                #     orbita.set_rpy_orientation(rpys[0])
                #     print('Orientation:', orbita.get_rpy_orientation())
                ###

                ### METHOD 2: Q
                # quaternions = []
                # if rotation_matrices:
                #     for R in rotation_matrices:
                #         theta = transformer.get_theta_from_R(R)
                #         q = transformer.get_q(R)
                #         theta_check = transformer.get_theta_from_q(q)
                #         if theta < SAFETY_RAD_LIMIT and theta > RAD_LOWER_LIMIT and abs(theta-theta_check) < THETA_ERR_LIMIT:
                #             quaternions.append(q)

                # if quaternions:
                #     orbita.set_orientation(quaternions[0])
                ###

        # orbita.go_to_sleep()

if __name__ == "__main__":
    main()
