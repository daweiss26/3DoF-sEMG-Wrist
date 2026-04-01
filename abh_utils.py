"""
Many of these functions are attributed to the work of Jesse Cornman.
These function serve as utilities for initializing and running the Ability Hand Mediapipe Demo.
"""
import numpy as np
import serial
from serial.tools import list_ports
from scipy import signal
import struct


class AbilityHandUtils:
	FRAME_CHAR = 0x7E
	ESC_CHAR = 0x7D
	MASK_CHAR = 0x20

	def get_serial_port(self, target_port: str):
		"""Connect to the desired serial port or return None if not found"""
		for port in list_ports.comports():
			if port.name.partition(' ')[0] == target_port:
				return serial.Serial(port.device, 921600, timeout=1)
		return None

	def py_sos_iir(self, newsample, w, sos):
		""" 
		Scipy style coefficients. For mult-order sections, scipy 
		absorbs all gains into the numerator of the last section, as 
		opposed to matlab which spreads it out. Therefore scipy style
		SOS has no gain associated
		""" 
		b = sos[0:3] # NUMERATOR
		a = sos[3:6] # DENOMINATOR
		w[2] = w[1]
		w[1] = w[0]
		w[0] = newsample * a[0] - a[1] * w[1] - a[2] * w[2]	# a0 is always 1
		fout = b[0] * w[0] + b[1] * w[1] + b[2] * w[2]
		return fout, w

	def get_low_pass_filter(self, fs: int = 30, wn: int = 3):
		"""
		Design the low pass filter we will use on the angle outputs.
		This smooths out the inputs to the hand.
		NOTE:
			1. fs is entered here. This is the estimated update frequency of our loop.
				might not be accurate! depends on our actual FPS. Hz
			2. Cutoff is entered here as Wn. Units are in Hz (bc. fs is in hz also)
			
			N must be 2 for this filter to be valid. N greater than 2 yields 2 sections, 
			and we're only doing 1 section.
			
		Higher values of Wn -> less aggressive filtering.
		Lower values of Wn -> more aggressive filtering.
		Wn must be greater than 0 and less than nyquist (Fs/2).
		"""
		# Consider wn=2 for smoother movement
		return signal.iirfilter(2, Wn=wn, btype='lowpass', analog=False, ftype='butter', output='sos', fs=fs)
	
	
	def farr_to_barr(self, addr, farr):
		"""
		Sends the array farr (which should have only 6 elements, or the hand won't do anything)
		Byte positions:
			0th: 0x50 
			1st: AD (control mode)
			payload: farr as the payload (4 bytes per value),
			last: checksum
		Must be 27 total bytes for the hand to do anything in response.
		"""
		barr = []
		barr.append((struct.pack('<B', addr))[0]) # Device ID
		barr.append((struct.pack('<B', 0xAD))[0]) # Control mode
		# Following block of code converts fpos into a floating point byte array and loads it into barr bytewise
		for fp in farr:
			b4 = struct.pack('<f',fp)
			for b in b4:
				barr.append(b)
		
		# Calculate the checksum and load it into the final byte
		barr.append(self.compute_checksum(barr))
		
		return barr

	def compute_checksum(self, barr):
		"""Prepares checksum"""
		sum = 0
		for b in barr:
			sum = sum + b
		chksum = (-sum) & 0xFF
		return chksum

	def ht_inverse(self, hin):
		"""
		Computes the inverse of the input matrix. Valid only for inputs which are homogeneous transformation matrices.
		Uses the property of rotation matrices where the inverse is equal to the transpose for fast computation. 
		"""
		hout = np.zeros((4,4))
		hout[0:3, 0:3] = np.transpose(hin[0:3,0:3])
		hout[0:3, 3] = -np.dot(hout[0:3,0:3], hin[0:3,3])
		return hout

	def v3_to_v4(self, vin):
		"""Pads a 1 to the end of an numpy 3-vector"""
		vout = np.zeros(4)
		vout[0:3] = vin
		vout[3] = 1
		return vout

	def vect_angle(self, v1, v2):
		"""
		Computes the angle between two n-dimensional vectors by 
		computing the arccos of the normalized dot product.
			
			NOTE: This will produce an incorrect result if a 1-padded
			vector (i.e. the type used for homogeneous transformation 
			matrix multiplication) is used. Be sure to call on vect[0:3]
			if using a 1-padded 3 vector.
		"""
		dp = np.dot(v1,v2)
		mp = np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2))
		cos_theta = np.clip(dp / mp, -1.0, 1.0)
		return np.arccos(cos_theta)

	def ht_from_2_vectors(self, vx, vyref, origin):
		"""Returns a 4x4 homogeneous transformation matrix"""
		ret = np.zeros((4,4))
		
		vz = np.cross(vx, vyref)
		vz = vz/self.mag(vz)
		vy = np.cross(vz, vx)
		vy = vy/self.mag(vy)
		ret[0:3, 0] = vx
		ret[0:3, 1] = vy
		ret[0:3, 2] = vz
		ret[0:3, 3] = origin
		ret[3, 0:4] = np.array([0,0,0,1])
		return ret

	def clamp(self, val, lowerlim, upperlim):
		"""Clamps value to lowerlim-upperlim"""
		return max(min(upperlim,val),lowerlim)

	def mag(self, v):
		"""Fast computation of the vector magnitude of input v"""
		return np.sqrt(v.dot(v))

	def linmap(self, v, p_out, p_in):
		"""Linear mapping helper function"""
		return (v-p_in[0])*((p_out[1]-p_out[0])/(p_in[1]-p_in[0]))

	def to_vect(self, v):
		"""Converts a mediapipe landmark_pb2 landmark to a numpy array 3-vector"""
		return np.array([v.x, v.y, v.z])
	
	def ppp_stuff(self, array: bytearray, create_copy=False) -> bytearray:
		"""Stuffing involves adding a FRAME_CHAR 0x7E '~' to the begining and end of
		a frame and XOR'ing any bytes with MASK_CHAR 0x20 that equal the FRAME/ESC
		char.  This allows you to determine the beginning and end of a frame and not
		have FRAME_CHAR or ESC_CHAR that are actually in the data confuse the parsing
		of the frame"""
		if create_copy:  # I'm fine always modifying original array
			array = array.copy()

		# Find ESC and FRAME chars
		ind = [i for i, v in enumerate(array) if v == self.ESC_CHAR or v == self.FRAME_CHAR]

		for i in ind:  # Mask Chars
			array[i] = array[i] ^ self.MASK_CHAR

		# Insert ESC char in front of masked char reverse to prevent index mess up
		for i in sorted(ind, reverse=True):
			array.insert(i, self.ESC_CHAR)

		array.insert(0, self.FRAME_CHAR)  # Mark beginning of frame
		array.append(self.FRAME_CHAR)  # Mark end of the frame

		return array
