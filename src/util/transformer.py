"""Helper class to transform between landmarks, rotation matrices, roll-pitch-yaw sets, and quaternions"""

import numpy as np


class Transformer:

    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        None

    ### Get theta (rotation magnitude)
    def get_theta_from_unit_vector(self, u1, u2):
        return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
    
    def get_theta_from_R(self, R):
        """Get total theta from rotation matrix"""
        return np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    
    def get_theta_from_q(self, q):
        return 2 * np.arccos(q[3])

    def get_theta_between_q(self, q1, q2):
        """Calculates the angular distance (in radians) between two quaternions. Angular distance = 2 * acos(|q1 . q2|)."""
        q1, q2 = np.array(q1), np.array(q2)
        dot_product = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0) 
        return 2 * np.arccos(dot_product)
    ###

    ### Get tilt (roll and pitch movement)
    def get_tilt_from_R(self, R):
        """Get tilt (exclude yaw/rotation) from rotation matrix"""
        return self.get_tilt_from_q(self.get_q_from_R(R))

    def get_tilt_from_rpy(self, roll, pitch):
        """Get tilt (exclude yaw/rotation) from roll and pitch"""
        return np.sqrt(roll**2 + pitch**2)

    def get_tilt_from_q(self, q):
        """Get tilt (exclude yaw/rotation) from quaternion"""
        q_norm = self.get_q_norm(q)
        q_tilt, _ = self.get_q_tilt_twist(q_norm)
        _, tilt_theta = self.get_axis_theta_from_q(q_tilt)
        return tilt_theta
    ###
    
    ### Get rotation matrix (R)
    def get_R_from_rpy(self, roll, pitch, yaw):
        """Convert from roll, pitch, yaw set to rotation matrix in ZYX form"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Z points out of the frame (roll axis)
        Rz = np.array([
            [cr,  -sr, 0.0],
            [sr,   cr, 0.0],
            [0.0, 0.0, 1.0]
        ])
        # Y points upward (yaw axis)
        Ry = np.array([
            [ cy, 0.0,  sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0,  cy]
        ])
        # X points to the side (pitch axis)
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0,  cp, -sp],
            [0.0,  sp,  cp]
        ])
        return Rz @ Ry @ Rx

    def get_R_from_q(self, q):
        """Converts a quaternion (qz (Roll), qx (Pitch), qy (Yaw), qw) to a 3x3 rotation matrix"""
        qz, qx, qy, qw = q
        return np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

    def get_R_from_landmarks(self, landmarking_result, mirror_hand=False):
        """Calculates the transformation matrix to turn the elbow-to-wrist vector into the wrist-to-hand vector"""
        elbow_to_wrist_unit, normal_unit, wrist_to_hand_unit, palm_vector_unit = self.get_coordinate_system(landmarking_result)

        elbow_to_wrist_cross_unit = np.cross(elbow_to_wrist_unit, normal_unit)
        wrist_to_hand_cross_unit = np.cross(wrist_to_hand_unit, palm_vector_unit)

        # These should be using ZYX Euler angle convention (yaw, pitch, roll).
        # Even though the palm vector actually corresponds to roll in our case,
        # and the vertical vector actually corresponds to yaw.
        # This convention allows us to use the formula for the transformation matrix.
        # We can swap around the angles after to suit our needs.
        X  = np.column_stack([normal_unit, elbow_to_wrist_unit, elbow_to_wrist_cross_unit])
        Xp = np.column_stack([palm_vector_unit, wrist_to_hand_unit, wrist_to_hand_cross_unit])
        R = X @ Xp.T

        # If you want to have a mirror effect on the output, transform the matrix by flipping the y and z axes
        if mirror_hand: R = np.diag([1.0, -1.0, -1.0]) @ R @ np.diag([1.0, -1.0, -1.0])
        return R
    ###
        
    ### Get roll, pitch, yaw (rpy)
    def wrap_pi(self, theta):
        return (theta + np.pi) % (2.0 * np.pi) - np.pi
    # The best way to get rpy from q is to convert to R first
    def get_rpy_from_R(self, R):
        """Turns the transformation matrix into roll-pitch-yaw angles"""
        if np.isclose(R[2, 0], 1.0, atol=0.05):
            yaw = -np.pi / 2.0
            pitch = 0.0
            roll = np.arctan2(-R[0, 1], R[1, 1])
            
        elif np.isclose(R[2, 0], -1.0, atol=0.05):
            yaw = np.pi / 2.0
            pitch = 0.0
            roll = np.arctan2(-R[0, 1], R[1, 1])
            
        else:
            yaw1 = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)) # Rotation about the y (vertical) axis, typically pitch, Orbita uses it as yaw
            pitch1 = np.arctan2(R[2, 1], R[2, 2]) # Rotation about the x (horizontal) axis, typically roll, Orbita uses it as pitch
            roll1 = np.arctan2(R[1, 0], R[0, 0]) # Rotation about the z (in/out) axis, typically yaw, Orbita uses it as roll

            yaw2 = np.pi - yaw1
            pitch2 = np.arctan2(-R[2, 1], -R[2, 2])
            roll2 = np.arctan2(-R[1, 0], -R[0, 0])

            sol1 = np.array([
                self.wrap_pi(roll1),
                self.wrap_pi(pitch1),
                self.wrap_pi(yaw1)
            ])
            sol2 = np.array([
                self.wrap_pi(roll2),
                self.wrap_pi(pitch2),
                self.wrap_pi(yaw2)
            ])

            candidates = [sol1, sol2]
            roll, pitch, yaw = min(candidates, key=lambda s: np.hypot(s[0], s[1]))

        return roll, pitch, yaw
    ###

    ### Get quaternion (q)
    def get_q_norm(self, q, eps=1e-12):
        q = np.array(q, dtype=float)
        n = np.linalg.norm(q)
        if n < eps:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / n

    # The best way to get q from rpy is to convert to R firstdef 
    def get_q_from_R(self, R):
        """Get quaternion from rotation matrix robustly, avoiding 180-deg singularities."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            
        # We want to allow continuous rotation and not lock q in the positive hemisphere
        # if qw < 0:
        #     qw, qx, qy, qz = -qw, -qx, -qy, -qz

        return (qz, qx, qy, qw) # Orbita rpy order
    ###

    ### Helper function for getting rotation matrix (R) from landmarks
    def get_coordinate_system(self, landmarking_result):
        """Get coordinate system for base of the hand"""
        landmarks, handedness = landmarking_result

        elbow = None
        if len(landmarks) == 23: elbow = np.array([self.width*landmarks[22].x, self.height*landmarks[22].y, self.width*landmarks[22].z])
        wrist = np.array([self.width*landmarks[0].x, self.height*landmarks[0].y, self.width*landmarks[0].z])
        hand = np.array([self.width*landmarks[21].x, self.height*landmarks[21].y, self.width*landmarks[21].z])
        hand2 = np.array([self.width*landmarks[5].x, self.height*landmarks[5].y, self.width*landmarks[5].z])

        if len(landmarks) == 23:
            elbow_to_wrist = wrist - elbow
            elbow_to_wrist[2] = 0
            elbow_to_wrist_unit = elbow_to_wrist / np.linalg.norm(elbow_to_wrist)
        else:
            elbow_to_wrist_unit = np.array([0,-1,0])

        wrist_to_hand = hand - wrist
        wrist_to_hand_unit = wrist_to_hand / np.linalg.norm(wrist_to_hand)

        wrist_to_hand2 = hand2 - wrist
        wrist_to_hand2_unit = wrist_to_hand2 / np.linalg.norm(wrist_to_hand2)
        palm_vector = np.cross(wrist_to_hand2_unit, wrist_to_hand_unit)
        if handedness == 'Left': palm_vector *= -1
        palm_vector_unit = palm_vector / np.linalg.norm(palm_vector)

        return elbow_to_wrist_unit, np.array([0,0,-1]), wrist_to_hand_unit, palm_vector_unit
    ###
    
    ### Helper functions for emd.py
    def get_hand_vector(self, landmarking_result):
        """Get vector from wrist to fingers"""
        landmarks, _ = landmarking_result
        wrist = np.array([self.width*landmarks[0].x, self.height*landmarks[0].y, self.width*landmarks[0].z])
        hand = np.array([self.width*landmarks[21].x, self.height*landmarks[21].y, self.width*landmarks[21].z])

        wrist_to_hand = hand - wrist
        wrist_to_hand_unit = wrist_to_hand / np.linalg.norm(wrist_to_hand)

        return wrist_to_hand_unit

    def get_distance(self, data, l2=False, transpose=False):
        data = np.array(data)
        if transpose:
            data = data.T
        diffs = np.diff(data, axis=0)
        if l2:
            return np.sqrt(np.sum(diffs**2, axis=1))
        else:
            return np.sum(np.abs(diffs), axis=1)

    def get_max(self, data):
        return np.max(data), np.argmax(data)
    ###

    ### Clamped quaternion
    def apply_hemisphere_guard(self, q_new, q_prev):
        """Ensure commands stay in the same hemisphere and prevent >180deg rotation"""
        if q_prev is None:
            return q_new

        if np.dot(q_new, q_prev) < 0:
            q_new = -q_new

        home = np.array([0, 0, 0, 1])
        dot_wrt_home = np.dot(q_new, home)
        
        # A negative dot product indicates a hemisphere flip
        if dot_wrt_home < 0.15:
            # Prevent movement by returning the last known good orientation
            return q_prev
        return q_new

    def get_q_inverse(self, q, eps=1e-8):
        q = np.array(q, dtype=float)
        denom = np.dot(q, q)
        if denom < eps:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return self.get_q_conjugate(q) / denom

    def get_q_conjugate(self, q):
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    def q_multiply(self, q1, q2):
        """Multiply quaternions in this codebase's order (qz, qx, qy, qw)"""
        z1, x1, y1, w1 = q1
        z2, x2, y2, w2 = q2

        # Standard quaternion multiplication
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2

        # Convert back to (qz, qx, qy, qw)
        return np.array([z, x, y, w])

    def get_q_from_tilt_twist(self, q_norm):
        """
        Decompose normalized q into (q_tilt * q_twist)
        q_twist: is the pure rotation about the global y-axis (yaw)
        q_tilt: rotation about roll and pitch
        """
        _, _, qy, qw = self.get_q_norm(q_norm)

        # Keep only y-axis (yaw) component in vector part
        q_twist = np.array([0.0, 0.0, qy, qw], dtype=float)
        q_twist = self.get_q_norm(q_twist)

        # Isolate q_tilt component by multiplying original by inverse of q_twist
        q_tilt = self.get_q_norm(self.q_multiply(self.get_q_inverse(q_twist), q_norm))

        return q_tilt, q_twist
    
    def get_q_from_axis_theta(self, axis, theta, eps=1e-8):
        """Converts and axis and an angle into a quaternion"""
        axis = np.array(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < eps:
            return np.array([0.0, 0.0, 0.0, 1.0])

        axis = axis / n
        s = np.sin(theta / 2.0)
        x = axis[0] * s
        y = axis[1] * s
        z = axis[2] * s
        w = np.cos(theta / 2.0)

        return np.array([z, x, y, w])

    def get_axis_theta_from_q(self, q_norm, eps=1e-8):
        """Breaks down normalized quaternion into axis in standard coordinates and angle"""
        qz, qx, qy, qw = self.get_q_norm(q_norm)
        qw = np.clip(qw, -1.0, 1.0)

        theta = 2.0 * np.arccos(qw)
        s = np.sqrt(max(1.0 - qw*qw, 0.0))

        if s < eps:
            return np.array([1.0, 0.0, 0.0]), 0.0

        axis = np.array([qx, qy, qz]) / s

        if theta > np.pi:
            theta = 2.0 * np.pi - theta
            axis = -axis

        return axis, theta
    
    def get_q_clamped_from_R(self, R, q_prev, max_tilt):
        """Clamp the tilt of a rotation to max_tilt (rad) while preserving yaw about the global y-axis (yaw)"""
        # Convert input rotation matrix to quaternion convention
        return self.get_q_clamped(self.get_q_from_R(R), q_prev, max_tilt)

    def get_q_clamped(self, q_curr, q_prev, max_tilt):
        """Clamp the tilt of a rotation to max_tilt (rad) while preserving yaw about the global y-axis (yaw)"""
        q_norm = self.get_q_norm(q_curr)

        # Decompose into tilt and yaw
        q_tilt, q_twist = self.get_q_from_tilt_twist(q_norm)

        # Measure current tilt magnitude
        tilt_axis, tilt_theta = self.get_axis_theta_from_q(q_tilt)
        if tilt_theta <= max_tilt:
            q_new = q_norm
        else:
            # Clamp only the tilt magnitude
            q_tilt_clamped = self.get_q_from_axis_theta(tilt_axis, max_tilt)
            # Recombine: same yaw, reduced tilt
            q_new = self.get_q_norm(self.q_multiply(q_twist, q_tilt_clamped))

        return self.apply_hemisphere_guard(q_new, q_prev)
    ###
