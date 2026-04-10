"""
Trains a CNN using data collected in an .npz file.
It takes in N samples that are dimension 125, 8 (timesteps, channels),
and it outputs dimension 3, 1 for 3D angular velocity.
80% is used as training, and 20% is used for testing.

Methodology:
1. TCN: Uses dilated, causal convolutions to capture long-range temporal dependencies .
2. Causal Padding: Ensures prediction at time t only uses data from t and before.
3. Incremental Mode: Allows loading a base model and adapting it to a new session (IRR concept).
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

INPUT_SHAPE = (125, 8) # 0.25s window @ 500Hz
BATCH_SIZE = 16
EPOCHS_FRESH = 60
EPOCHS_FINE_TUNE = 10
LR_FRESH = 0.000001
LR_FINE_TUNE = 0.000001


def plot_learning_curves(history):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 28,             # Base font size scaled for poster readability
        'axes.labelsize': 28,        # Larger axis labels
        'axes.titlesize': 26,        # Prominent title
        'axes.titleweight': 'bold',
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 18,
        'figure.dpi': 100,           # 300 DPI for crisp physical printing
        'axes.linewidth': 3,          # Thicker axis borders
        'xtick.major.size': 8,  # length in points
        'xtick.major.width': 2,   # thickness in points
        'ytick.major.size': 8,
        'ytick.major.width': 2,
        'xtick.minor.size': 4,
        'xtick.minor.width': 1,
        'ytick.minor.size': 4,
        'ytick.minor.width': 1
    })

    # Initialize figure with an appropriate aspect ratio
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the validation loss data
    ax.plot(history.history['val_loss'], 
            color='#D32F2F',         # Professional, deep red
            linewidth=5.0,           # Increased thickness for high visibility
            label='Val Loss', 
            zorder=3)                # Ensures the line is drawn above the grid

    # Professional formatting
    # ax.set_title('TCN Model Loss Against Validation Data')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')

    # Add a subtle grid to help the audience read values from a distance
    ax.grid(True, which='major', color='#E0E0E0', linestyle='-', linewidth=2.0, zorder=1)
    ax.grid(True, which='minor', color='#F5F5F5', linestyle=':', linewidth=1.5, zorder=1)
    ax.minorticks_on()

    # Clean up the top and right borders (spines) for a modern look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the high-res image directly in addition to showing it
    plt.tight_layout() # Ensures labels are not cut off when saved
    plt.savefig('val_loss_poster_2.png', format='png', dpi=100, bbox_inches='tight')

    # plt.figure(figsize=(10, 5))

    # Loss
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Val Loss')
    # plt.title('TCN Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE')
    # plt.legend()

    # MAE
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['mae'], label='Train MAE')
    # plt.plot(history.history['val_mae'], label='Val MAE')
    # plt.title('TCN Model MAE')
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE (rad/s)')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

def residual_block(x, filters, kernel_size, dilation_rate):
    """Creates a TCN Residual Block: [Dilated Conv -> Norm -> Relu -> Dropout] + Shortcut -> Output"""
    prev_x = x

    # Calculate manual causal padding: (kernel_size - 1) * dilation_rate
    padding_size = (kernel_size - 1) * dilation_rate

    # Prepend zeros to the beginning of the sequence (left-side padding)
    # The tuple (padding_size, 0) adds to the start and 0 to the end
    x = layers.ZeroPadding1D(padding=(padding_size, 0))(x)

    # Dilated Causal Separable Convolution
    x = layers.SeparableConv1D(filters=filters, 
                               kernel_size=kernel_size, 
                               dilation_rate=dilation_rate, 
                               padding='valid',
                               activation='relu')(x)
    x = layers.BatchNormalization()(x) # Stabilizes learning
    x = layers.SpatialDropout1D(0.1)(x) # Drops entire feature maps

    # Shortcut
    if prev_x.shape[-1] != filters: # They must have the same number of filters (dimensions)
        shortcut = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(prev_x) # Use a 1x1 Convolution to "project" old to new
    else:
        shortcut = prev_x

    res_x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(res_x)

def build_model(input_dim):
    """Builds a TCN with causal dilated convolutions."""
    inputs = layers.Input(shape=input_dim)
    x = layers.Conv1D(32, kernel_size=3, padding='causal', activation='relu')(inputs)
    
    # Dilated Convolutions Filters double as we go deeper to capture more complex abstract patterns
    # Receptive Field: (1 + sum((kernel-1) * dilation)
    x = residual_block(x, filters=64, kernel_size=3, dilation_rate=1)
    x = residual_block(x, filters=64, kernel_size=3, dilation_rate=2)
    x = residual_block(x, filters=64, kernel_size=3, dilation_rate=4)
    x = residual_block(x, filters=128, kernel_size=3, dilation_rate=8)
    x = residual_block(x, filters=128, kernel_size=3, dilation_rate=16)
    
    # Flattening preserves temporal data
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x) # Prevent overfitting
    outputs = layers.Dense(3)(x) # 3D Angular Velocity (rad/s)
    
    return models.Model(inputs=inputs, outputs=outputs)

def load_data(npz_file):
    data = np.load(npz_file)
    return data['emg'], data['velocity']

def main():
    parser = argparse.ArgumentParser(description="Train or Fine-tune Model")
    parser.add_argument("input_file", help="Path to training data (.npz)")
    parser.add_argument("model_name", help="Name for the output model file")
    parser.add_argument("--resume", help="Path to existing .keras model to fine-tune", default=None)
    args = parser.parse_args()
    _, input_file = args.input_file.split('=', 1)
    _, model_name = args.model_name.split('=', 1)

    # This data should already be normalized per session
    print(f"Loading data from {input_file}...")
    X, y = load_data(input_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    if args.resume:
        # Lower learning rate and fewer epochs for fine-tuning
        print(f"INCREMENTAL MODE: Fine-tuning {args.resume} into {model_name}")
        model = tf.keras.models.load_model(args.resume)
        optimizer = optimizers.Adam(learning_rate=LR_FINE_TUNE)
        epochs = EPOCHS_FINE_TUNE
    else:
        print(f"INITIALIZE MODE: Building {model_name} from scratch")
        model = build_model(INPUT_SHAPE)
        optimizer = optimizers.Adam(learning_rate=LR_FRESH)
        epochs = EPOCHS_FRESH

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{model_name}.keras", save_best_only=True, monitor='val_loss', mode='min')
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint]
    )
    # Visualize how well the model did
    plot_learning_curves(history)

if __name__ == "__main__":
    main()
