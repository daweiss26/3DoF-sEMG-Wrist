import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from itertools import product, combinations

class QuaternionVisualizer:
    def __init__(self):
        """Initializes a real-time 3D plot with a fixed cube and moving orientation vector."""
        plt.ion() # Turn on interactive mode
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Vertices of a cube centered at 0,0,0
        r = [-1, 1]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            # Only draw edges where points differ by exactly one coordinate
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                self.ax.plot3D(*zip(s, e), color="lightgray", linestyle="--")
                
        # --- 2. Setup Plot Limits and Camera ---
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-0.25, 1.5])
        self.ax.set_zlim([-1.5, 1.5])
        self.ax.set_xlabel('Front (Roll/qz)')
        self.ax.set_ylabel('Side (Pitch/qx)')
        self.ax.set_zlabel('Up (Yaw/qy)')
        
        # Set viewing angle
        self.ax.view_init(elev=20, azim=45) 
        
        # Init vector
        self.vector_line, = self.ax.plot([], [], [], color='red', linewidth=3, marker='o')
        
        plt.show(block=False)

    def update(self, q: Quaternion):
        """Rotates a base vector by the quaternion and updates the line."""
        # Initialize vector (straight up in yaw direction)
        base_vector = np.array([0.0, 0.0, 1.0])
        
        # Rotate the base vector by the quaternion
        rotated_vector = q.rotate(base_vector)
        
        # Update the line to go from Origin (0,0,0) to the rotated vector
        # Updating the line is much faster than redrawing it entirely
        # So we just set the data instead of deleting and adding
        self.vector_line.set_data([0, rotated_vector[2]], [0, rotated_vector[0]]) # qz and qx
        self.vector_line.set_3d_properties([0, rotated_vector[1]]) # qy

        # Flush the GUI events to draw the new frame immediately
        self.ax.draw_artist(self.vector_line) # Should be faster than self.fig.canvas.draw()...
        self.fig.canvas.flush_events()
