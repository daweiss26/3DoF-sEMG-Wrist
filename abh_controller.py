"""Controller for the Psyonic Ability Hand"""

from enum import IntEnum
import numpy as np
from abh_utils import AbilityHandUtils

class HAND_LANDMARK_MAP(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class AbilityHand:

	def __init__(self, target_port, errorOnConnectionFailure=False):
		self.utils = AbilityHandUtils()
		self.outp_fng = [5, 100]
		self.inp_fng = [20, 130]
		self.outrange_fng = [5, 110]

		self.outp_tr = [-5, -70]
		self.inp_tr = [-50, -75]
		self.outrange_tr = [-500, -5]

		self.outp_tf = [0, 60]
		self.inp_tf = [15,-40]
		self.outrange_tf = [10, 90]

		self.filter_fpos = True
		
		# Initialize array used for writing out hand positions
		self.fpos = [15, 15, 15, 15, 15, -15]
		
		# Initialize array used for filtering
		self.warr = []
		for f in self.fpos:
			self.warr.append([0,0,0])
		
		self.hw_b = np.zeros((4,4))
		self.hb_w = np.zeros((4,4))
		self.hb_ip = np.zeros((4,4))
		self.hip_b = np.zeros((4,4))
		
		self.handed_sign = -1 # 1 for left hand, -1 for right hand
		self.scale = 1

		# Get low-pass filter
		self.lpf_sos = self.utils.get_low_pass_filter()

		# Get serial connection
		self.serial = self.utils.get_serial_port(target_port)
		if self.serial == None:
			if errorOnConnectionFailure:
				raise NameError(f'Error initializing using port: {target_port}')
			else:
				print(f'Error initializing using port: {target_port}, continuing without writing')

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.relax()
		return
	
	def update(self, landmarking_result):
		self.get_new_fpos(landmarking_result)
		self.send_command()
	
	def get_new_fpos(self, landmarking_result):
		# Extract landmarks
		landmarks = landmarking_result[0][:21]
		hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
		handedness = landmarking_result[1]
		if handedness == 'Left':
			self.handed_sign = 1
		elif handedness == 'Right':
			self.handed_sign = -1

		# Get scale
		mcp_indices = [HAND_LANDMARK_MAP.INDEX_FINGER_MCP, HAND_LANDMARK_MAP.MIDDLE_FINGER_MCP, HAND_LANDMARK_MAP.RING_FINGER_MCP, HAND_LANDMARK_MAP.PINKY_MCP]
		mcp_coords = hand_landmarks[mcp_indices]
		self.scale = np.linalg.norm(np.diff(mcp_coords, axis=0), axis=1).mean()

		# Construct coordinate frame
		wrist = hand_landmarks[HAND_LANDMARK_MAP.WRIST]
		index_mcp = hand_landmarks[HAND_LANDMARK_MAP.INDEX_FINGER_MCP]
		pinky_mcp = hand_landmarks[HAND_LANDMARK_MAP.PINKY_MCP]

		vx = index_mcp - wrist
		vx /= np.linalg.norm(vx)
		
		vy_ref = pinky_mcp - wrist
		vz = np.cross(vx, vy_ref)
		vz /= np.linalg.norm(vz)

		vy = np.cross(vz, vx)
		vy /= np.linalg.norm(vy)

		self.hw_b = np.eye(4)
		self.hw_b[0:3, 0] = vx
		self.hw_b[0:3, 1] = vy
		self.hw_b[0:3, 2] = vz
		self.hw_b[0:3, 3] = wrist
		self.hb_w = self.utils.ht_inverse(self.hw_b)

		# Transform all points to base frame
		target_indices = [
			HAND_LANDMARK_MAP.THUMB_TIP, HAND_LANDMARK_MAP.THUMB_IP, HAND_LANDMARK_MAP.THUMB_MCP, HAND_LANDMARK_MAP.THUMB_CMC,
			HAND_LANDMARK_MAP.INDEX_FINGER_TIP, HAND_LANDMARK_MAP.INDEX_FINGER_PIP, HAND_LANDMARK_MAP.INDEX_FINGER_MCP,
			HAND_LANDMARK_MAP.MIDDLE_FINGER_TIP, HAND_LANDMARK_MAP.MIDDLE_FINGER_PIP, HAND_LANDMARK_MAP.MIDDLE_FINGER_MCP,
			HAND_LANDMARK_MAP.RING_FINGER_TIP, HAND_LANDMARK_MAP.RING_FINGER_PIP, HAND_LANDMARK_MAP.RING_FINGER_MCP,
			HAND_LANDMARK_MAP.PINKY_TIP, HAND_LANDMARK_MAP.PINKY_PIP, HAND_LANDMARK_MAP.PINKY_MCP
		]

		points = hand_landmarks[target_indices]
		points_h = np.hstack([points, np.ones((len(points), 1))])
		all_b = (self.hb_w @ points_h.T).T[:, :3] / self.scale

		# Thumb
		thumb_tip_b, thumb_ip_b, thumb_mcp_b, thumb_cmc_b = all_b[0:4]
		ang_tr = np.degrees(np.arctan2(self.handed_sign * thumb_ip_b[2], -thumb_ip_b[1]))
		self.fpos[5] = self.utils.clamp(
			self.utils.linmap(ang_tr, self.outp_tr, self.inp_tr), 
			*self.outrange_tr
		)
		
		vx_f = thumb_ip_b - thumb_mcp_b
		vx_f /= np.linalg.norm(vx_f)
		self.hb_ip = self.utils.ht_from_2_vectors(vx_f, thumb_cmc_b, thumb_ip_b)
		self.hip_b = self.utils.ht_inverse(self.hb_ip)
		thumb_tip_ip = self.hip_b @ np.append(thumb_tip_b, 1.0)
		ang_tf = np.degrees(np.arctan2(thumb_tip_ip[1], thumb_tip_ip[0]))
		self.fpos[4] = self.utils.linmap(ang_tf, self.outp_tf, self.inp_tf)

		# Reshape the remaining points into (4 fingers, 3 joints, 3 coords)
		finger_data = all_b[4:].reshape(4, 3, 3) 
		for i in range(4):
			tip_b, pip_b, mcp_b = finger_data[i]

			# Flexion angles
			v_tip_pip = tip_b - pip_b
			v_pip_mcp = pip_b - mcp_b
			
			# Use utils for angle logic but pass the 3D slices directly
			q1 = self.utils.vect_angle(v_tip_pip, v_pip_mcp)
			q2 = self.utils.vect_angle(v_pip_mcp, mcp_b)
			fng_ang = np.degrees(q1 + q2)
			self.fpos[i] = self.utils.linmap(fng_ang, self.outp_fng, self.inp_fng)

		# Filtering
		if self.filter_fpos:
			for i in range(len(self.fpos)):
				self.fpos[i], self.warr[i] = self.utils.py_sos_iir(self.fpos[i], self.warr[i], self.lpf_sos[0])

	def send_command(self):
		msg = self.utils.farr_to_barr(0x50, self.fpos)
		msg = self.utils.ppp_stuff(msg)
		if self.serial:
			self.serial.write(msg)

	def relax(self):
		msg = self.utils.farr_to_barr(0x50, [15, 15, 15, 15, 15, -15])
		msg = self.utils.ppp_stuff(msg)
		if self.serial:
			self.serial.write(msg)
