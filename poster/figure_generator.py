import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# # Reading the CSV file
df = pd.read_csv("./poster/Test1.csv", sep="\t")
emg = df.loc[:, 'FilteredChannel1']

def generate_pipeline_graphic():
    """Generates the 1000x400 graphic for the Sliding Window Pipeline"""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100, facecolor='white')
    ax.set_xlim(-10, 1000)
    ax.set_ylim(-10, 400)
    ax.axis('off')

    # --- 1. Raw sEMG Waveform ---
    t = np.linspace(0, 400, 300)
    # Generate synthetic sEMG signal
    # signal = (np.sin(t * 0.05) * np.sin(t * 0.2) * 20) + np.random.normal(0, 5, len(t)) # simulation
    signal = emg[100:400]
    
    # Plot top raw signal
    y_offset_raw = 320
    ax.plot(t, signal/13 + y_offset_raw, color='darkgray', linewidth=2.5)
    ax.text(200, 365, "Raw sEMG Stream (400ms)", ha='center', fontsize=12, fontweight='bold', color='black')

    # --- 2. Sliding Windows ---
    window_colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
    window_labels = ['0 - 250ms', '50 - 300ms', '100 - 350ms', '150 - 400ms']
    y_offsets = [230, 170, 110, 50]
    
    for i in range(4):
        start_ms = i * 50
        end_ms = start_ms + 250
        
        # Find indices for the slice
        idx_start = int((start_ms / 400) * 300)
        idx_end = int((end_ms / 400) * 300)
        
        t_slice = t[idx_start:idx_end]
        sig_slice = signal[idx_start:idx_end] / 13
        
        # Plot the window segment
        ax.plot(t_slice, sig_slice + y_offsets[i], color=window_colors[i], linewidth=2.5)
        
        # Add background highlight box
        rect = patches.Rectangle((start_ms, y_offsets[i]-25), 250, 50, linewidth=1, 
                                 edgecolor=window_colors[i], facecolor=window_colors[i], alpha=0.1)
        ax.add_patch(rect)
        
        # Label the window
        ax.text(start_ms - 10, y_offsets[i], window_labels[i], ha='right', va='center', 
                fontsize=10, fontweight='bold', color=window_colors[i])

    plt.tight_layout(pad=0)
    plt.savefig('poster/data_pipeline.png', dpi=300, transparent=True)
    plt.close()

# Execute generators
generate_pipeline_graphic()
print("Successfully generated")