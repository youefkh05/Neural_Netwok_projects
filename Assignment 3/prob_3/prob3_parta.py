import os
import tensorflow as tf
import warnings
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations warning

# Suppress TensorFlow deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
tf.get_logger().setLevel('ERROR')

# Create output folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs_parta")
os.makedirs(OUT_DIR, exist_ok=True)

# Spatial Attention Layer
class SpatialAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Data reduction (Ex: using only the first 10,000 images)
x_train = x_train[:10000]
y_train = y_train[:10000]

# Create both models
def create_baseline_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(4, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def create_attention_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(4, (3, 3), activation='relu'),
        SpatialAttention(),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        SpatialAttention(),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Train and evaluate models
def train_and_display(model, model_name):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"\n\033[1m=== Training {model_name} Model ===\033[0m")

    start_time = time.time()  # Start time

    history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                        validation_data=(x_test, y_test), verbose=1)

    end_time = time.time()  # End time
    training_time = end_time - start_time
    model.training_time = training_time
    model.history = history

    # Display training metrics
    print(f"""
    {model_name} Model Training Results:
    ====================================
    Final Training Accuracy:   {history.history['accuracy'][-1]:.4f}
    Final Test Accuracy:       {history.history['val_accuracy'][-1]:.4f}
    Final Training Loss:       {history.history['loss'][-1]:.4f}
    Final Test Loss:           {history.history['val_loss'][-1]:.4f}
    Total Training Time:       {training_time:.2f} seconds
    """)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='black', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Test Accuracy', color='red', linewidth=2)
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='black', linewidth=2)
    plt.plot(history.history['val_loss'], label='Test Loss', color='red', linewidth=2)
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{model_name}_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Display model summary
    print(f"\n\033[1m{model_name} Model Summary:\033[0m")
    model.summary()

    return model

# Create and train both models
baseline_model = create_baseline_model()
attention_model = create_attention_model()

baseline_model = train_and_display(baseline_model, "Baseline")
attention_model = train_and_display(attention_model, "Attention")

# Compare final results
test_loss_baseline, test_acc_baseline = baseline_model.evaluate(x_test, y_test, verbose=0)
test_loss_attention, test_acc_attention = attention_model.evaluate(x_test, y_test, verbose=0)

# Extract training metrics from history
train_acc_baseline = baseline_model.history.history['accuracy'][-1]
train_acc_attention = attention_model.history.history['accuracy'][-1]
train_loss_baseline = baseline_model.history.history['loss'][-1]
train_loss_attention = attention_model.history.history['loss'][-1]

print(f"""
{'='*95}
COMPREHENSIVE MODEL COMPARISON TABLE
{'='*95}

{'Metric':<30} {'Baseline':<20} {'Attention':<20} {'Difference':<20}
{'-'*95}
Train Accuracy (%)        {train_acc_baseline*100:<20.2f} {train_acc_attention*100:<20.2f} {(train_acc_attention-train_acc_baseline)*100:>+19.2f}
Test Accuracy (%)         {test_acc_baseline*100:<20.2f} {test_acc_attention*100:<20.2f} {(test_acc_attention-test_acc_baseline)*100:>+19.2f}
Train Loss                {train_loss_baseline:<20.4f} {train_loss_attention:<20.4f} {(train_loss_baseline-train_loss_attention):>+19.4f}
Test Loss                 {test_loss_baseline:<20.4f} {test_loss_attention:<20.4f} {(test_loss_baseline-test_loss_attention):>+19.4f}
Training Time (seconds)   {baseline_model.training_time:<20.2f} {attention_model.training_time:<20.2f} {(attention_model.training_time-baseline_model.training_time):>+19.2f}
{'='*95}

📊 KEY OBSERVATIONS:
  ✓ Train vs Test Accuracy Gap (Baseline):  {(train_acc_baseline-test_acc_baseline)*100:.2f}% (overfitting indicator)
  ✓ Train vs Test Accuracy Gap (Attention): {(train_acc_attention-test_acc_attention)*100:.2f}% (overfitting indicator)
  
  Winner in Test Accuracy:  {'Baseline' if test_acc_baseline > test_acc_attention else 'Attention'} 
                            ({abs(test_acc_baseline - test_acc_attention)*100:.2f}% difference)
  
  Winner in Training Time:  {'Baseline' if baseline_model.training_time < attention_model.training_time else 'Attention'}
                            ({abs(baseline_model.training_time - attention_model.training_time):.2f}s difference)

{'='*95}
""")

# Display attention visualization
def visualize_attention(model, digit_label, sample_idx=None):
    attention_outputs = [layer.output for layer in model.layers if 'spatial_attention' in layer.name]
    if not attention_outputs:
        print("No attention layers found in this model")
        return

    vis_model = keras.Model(inputs=model.inputs, outputs=attention_outputs)
    
    if sample_idx is None:
        digit_indices = np.where(y_test == digit_label)[0][:3]
    else:
        digit_indices = sample_idx if isinstance(sample_idx, list) else [sample_idx]
    
    fig, axes = plt.subplots(len(digit_indices), 3, figsize=(16, 6*len(digit_indices)))
    if len(digit_indices) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Spatial Attention Maps (channel-averaged) - Digit {digit_label}', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    for row, idx in enumerate(digit_indices):
        sample = x_test[idx].reshape(1, 28, 28, 1)
        attention_maps = vis_model.predict(sample, verbose=0)
        
        # Column 0: Input image
        axes[row, 0].imshow(sample[0, :, :, 0], cmap='gray', interpolation='nearest')
        axes[row, 0].set_title(f'Input (label={digit_label})', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Column 1: Attention Block 1
        attn_map1 = np.mean(attention_maps[0][0], axis=-1)
        im1 = axes[row, 1].imshow(attn_map1, cmap='hot', interpolation='bilinear')
        axes[row, 1].set_title('Attn Block 1', fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        
        # Column 2: Attention Block 2
        attn_map2 = np.mean(attention_maps[1][0], axis=-1)
        im2 = axes[row, 2].imshow(attn_map2, cmap='hot', interpolation='bilinear')
        axes[row, 2].set_title('Attn Block 2', fontsize=11, fontweight='bold')
        axes[row, 2].axis('off')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

print("\n[*] Saving Attention Visualization for Digit 0...")
fig_0 = visualize_attention(attention_model, digit_label=0)
plt.savefig(os.path.join(OUT_DIR, 'digit_0.png'), dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: {os.path.join(OUT_DIR, 'digit_0.png')}")
plt.close()

print("[*] Saving Attention Visualization for Digit 2...")
fig_2 = visualize_attention(attention_model, digit_label=2)
plt.savefig(os.path.join(OUT_DIR, 'digit_2.png'), dpi=300, bbox_inches='tight')
print(f"    ✓ Saved: {os.path.join(OUT_DIR, 'digit_2.png')}")
plt.close()