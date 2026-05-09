import os
import cv2
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time


# =========================
# CONFIG
# =========================
CACHE_DIR = "cache"
OUTPUT_DIR = "Figures"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CALLBACK TO VISUALIZE PREDICTIONS DURING TRAINING
# =========================
# =========================
# VAE Visualization Callback
# =========================
class VAEVisualizationCallback(tf.keras.callbacks.Callback):

    def __init__(self, decoder, latent_dim=20, interval=20, run_id=1):
        super().__init__()

        self.decoder = decoder
        self.latent_dim = latent_dim
        self.interval = interval
        self.run_id = run_id

    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.interval != 0:
            return

        plt.figure(figsize=(10,4))

        for digit in range(10):

            z = np.random.normal(0,0.3,(1,self.latent_dim))

            label = tf.keras.utils.to_categorical([digit], 10)

            img = self.decoder.predict(
                [z, label],
                verbose=0
            )[0]

            plt.subplot(2,5,digit+1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(str(digit))
            plt.axis('off')

        plt.tight_layout()

        path = os.path.join(
            OUTPUT_DIR,
            f"vae_run_{self.run_id}_epoch_{epoch+1}.png"
        )

        plt.savefig(path, dpi=300)
        plt.close()

        print(f"[INFO] Saved VAE visualization: {path}")
                         
# =========================
# SAVE / LOAD CACHE
# =========================
def save_cache(name, data):
    path = os.path.join(CACHE_DIR, name)

    np.save(path, data, allow_pickle=True)  

    print(f"[INFO] Saved cache: {path}")
    
def load_cache(name):
    path = os.path.join(CACHE_DIR, name)

    if os.path.exists(path):
        try:
            print(f"[INFO] Loaded cache: {path}")
            data = np.load(path, allow_pickle=True)

            #  convert numpy object → dict
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()

            return data

        except Exception:
            print(f"[WARNING] Corrupted cache → deleting {path}")
            os.remove(path)
            return None

    return None

# =========================
# LOAD MNIST
# =========================
def load_mnist():
    train_data = load_cache("train_data.npy")
    test_data  = load_cache("test_data.npy")

    if train_data is not None and test_data is not None:
        return (
            train_data["x"], train_data["y"],
            test_data["x"], test_data["y"]
        )

    print("[INFO] Downloading MNIST...")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
 
    # KEEP in [0,1] for VAE
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    save_cache("train_data.npy", {"x": x_train, "y": y_train})
    save_cache("test_data.npy", {"x": x_test, "y": y_test})

    return x_train, y_train, x_test, y_test

# =========================
# Shuffle dataset 
# =========================
def shuffle_dataset(x, y):
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]

# =========================
# Save figure
# =========================
def save_images_grid(images, labels, aug_types=None, filename="grid.png", n=10):    # Convert single image → batch
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        labels = np.array([labels])

    plt.figure(figsize=(10, 2))

    n = min(n, len(images))

    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')

        if aug_types is not None:
            title = f"{labels[i]}\n{aug_types[i][:25]}..."
        else:
            title = str(labels[i])

        plt.title(title, fontsize=8)
        plt.axis('off')

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved grid to: {path}")

# =========================
# cache dataset
# =========================
def get_reduced_dataset(x, y, samples_per_digit, cache_name=None):
    if cache_name:
        cached = load_cache(cache_name)

        if cached is not None:
            return cached["x"], cached["y"]

    x_small = []
    y_small = []

    for digit in range(10):
        idx = np.where(y == digit)[0][:samples_per_digit]
        x_small.append(x[idx])
        y_small.append(y[idx])

    x_small = np.concatenate(x_small)
    y_small = np.concatenate(y_small)

    print(f"[INFO] Reduced dataset: {x_small.shape}")

    if cache_name:
        save_cache(cache_name, {"x": x_small, "y": y_small})

    return x_small, y_small


# =========================
# AUGMENT DATASET (with cache + types)
# =========================
def augment_dataset(x, y, factor=10, cache_name=None):

    if cache_name:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached["x"], cached["y"], cached["aug_type"]

    print("[INFO] Augmenting dataset...")

    x_aug = []
    y_aug = []
    aug_types = []

    for i in range(len(x)):
        img = x[i]
        label = y[i]

        for _ in range(factor):

            aug_desc = []

            # -------------------------
            # ROTATION
            # -------------------------
            angle = np.random.uniform(-8, 8)
            M = cv2.getRotationMatrix2D((14,14), angle, 1)
            rotated = cv2.warpAffine(img.squeeze(), M, (28,28))
            aug_desc.append(f"rot({angle:.1f})")

            # -------------------------
            # SHIFT
            # -------------------------
            tx = np.random.randint(-2, 3)
            ty = np.random.randint(-2, 3)
            M_shift = np.float32([[1,0,tx],[0,1,ty]])
            shifted = cv2.warpAffine(rotated, M_shift, (28,28))
            aug_desc.append(f"shift({tx},{ty})")

            # -------------------------
            # SCALE
            # -------------------------
            scale = np.random.uniform(0.9, 1.1)
            M_scale = cv2.getRotationMatrix2D((14,14), 0, scale)
            scaled = cv2.warpAffine(shifted, M_scale, (28,28))
            aug_desc.append(f"scale({scale:.2f})")

            # -------------------------
            # NOISE
            # -------------------------
            noise = np.random.normal(0, 0.01, (28,28))
            final = np.clip(scaled + noise, 0, 1)
            aug_desc.append("noise")

            # Save
            x_aug.append(final[..., None])
            y_aug.append(label)
            aug_types.append(" | ".join(aug_desc))

    x_aug = np.array(x_aug)
    y_aug = np.array(y_aug)
    aug_types = np.array(aug_types)

    print(f"[INFO] Augmented dataset: {x_aug.shape}")

    if cache_name:
        save_cache(cache_name, {
            "x": x_aug,
            "y": y_aug,
            "aug_type": aug_types
        })

    return x_aug, y_aug, aug_types


# =========================
# Build a simple LeNet-like CNN
# =========================
def build_lenet():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28,28,1)),  # fix warning too

        tf.keras.layers.Conv2D(6, kernel_size=5, activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(16, kernel_size=5, activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# =========================
# Plot Results from CSV
# =========================
def plot_results_from_csv(csv_path):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8,6))

    plt.bar(
        df["Experiment"],
        df["Accuracy"]
    )

    plt.ylabel("Accuracy")
    plt.title("VAE Synthetic Data Results")

    plt.xticks(rotation=20)

    path = os.path.join(
        OUTPUT_DIR,
        "vae_results_plot.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved plot: {path}")

# =========================
#  Build Encoder (for VAE)
# =========================
def build_encoder(latent_dim=20):

    x_input = tf.keras.Input(shape=(28,28,1))
    y_input = tf.keras.Input(shape=(10,))

    # label map
    y_map = tf.keras.layers.Dense(28*28)(y_input)
    y_map = tf.keras.layers.Reshape((28,28,1))(y_map)

    x = tf.keras.layers.Concatenate()([x_input, y_map])

    # -------------------------
    # Conv blocks
    # -------------------------
    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        strides=2,
        padding='same'
    )(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        strides=2,
        padding='same'
    )(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)

    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)

    return tf.keras.Model(
        [x_input, y_input],
        [z_mean, z_log_var]
    )
  
# =========================
# Build Sampling Layer (for VAE)
# =========================
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon  

# =========================
# Build Decoder (for VAE)
# =========================
def build_decoder(latent_dim=20):

    z_input = tf.keras.Input(shape=(latent_dim,))
    y_input = tf.keras.Input(shape=(10,))

    x = tf.keras.layers.Concatenate()([z_input, y_input])

    # -------------------------
    # Dense projection
    # -------------------------
    x = tf.keras.layers.Dense(7 * 7 * 64)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Reshape((7,7,64))(x)

    # -------------------------
    # Upsample 7x7 -> 14x14
    # -------------------------
    x = tf.keras.layers.Conv2DTranspose(
        64,
        kernel_size=3,
        strides=2,
        padding='same'
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # -------------------------
    # Upsample 14x14 -> 28x28
    # -------------------------
    x = tf.keras.layers.Conv2DTranspose(
        32,
        kernel_size=3,
        strides=2,
        padding='same'
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # -------------------------
    # Final image
    # -------------------------
    x = tf.keras.layers.Conv2D(
        1,
        kernel_size=3,
        padding='same',
        activation='sigmoid'
    )(x)

    return tf.keras.Model([z_input, y_input], x)

        
# =========================
# VAE Model Class
# =========================
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        (x, y), _ = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder([x, y])
            z = self.sampling(z_mean, z_log_var)
            reconstruction = self.decoder([z, y])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + 0.0005 * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# =========================
# Train classifier with real + generated data
# ========================= 
def train_classifier_with_generated(
    x_real,
    y_real,
    x_fake,
    y_fake,
    x_test,
    y_test,
    name="set"
):    

    x_final = np.concatenate([x_real, x_fake])
    y_final = np.concatenate([y_real, y_fake])

    idx = np.random.permutation(len(x_final))

    x_final = x_final[idx]
    y_final = y_final[idx]

    model = build_lenet()

    history = model.fit(
        x_final,
        y_final,
        epochs=30,
        batch_size=64,
        verbose=1
    )

    _, acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"[RESULT] {name} Accuracy = {acc*100:.2f}%")

    return acc       

# =========================
# MAIN FUNCTION (VAE PIPELINE)
# =========================
def main():

    print("===== VAE PIPELINE START =====")

    # =========================
    # LOAD DATA (cached)
    # =========================
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"[INFO] Train shape: {x_train.shape}")

    # =========================
    # QUICK VISUAL CHECK
    # =========================
    save_images_grid(
        x_train[:10],
        y_train[:10],
        filename="MNIST_samples.png",
        n=10
    )
    

    # =========================
    # REDUCE DATASET (350 per digit)
    # =========================
    real_n = 350

    x_small, y_small = get_reduced_dataset(
        x_train, y_train,
        real_n,
        cache_name=f"real_{real_n}.npy"
    )

    print(f"[INFO] Reduced dataset: {x_small.shape}")

    # =========================
    # AUGMENT DATA (Step 1)
    # =========================
    x_aug, y_aug, aug_types = augment_dataset(
        x_small,
        y_small,
        factor=10,
        cache_name=f"aug_vae_{real_n}.npy"
    )
    
    # Visualize some augmented samples
    save_images_grid(
        x_aug[:5],
        y_aug[:5],
        aug_types=aug_types[:5],
        filename="vae_aug_samples.png",
        n=10
    )

    # Combine real + augmented
    x_vae = np.concatenate([x_small, x_aug])
    y_vae = np.concatenate([y_small, y_aug])

    print(f"[INFO] VAE training dataset: {x_vae.shape}")


    # =========================
    # TRAIN OR LOAD VAE
    # =========================
    latent_dim = 8
    
    y_vae_onehot = tf.keras.utils.to_categorical(y_vae, 10)
      
        
    # =========================
    # GENERATE SYNTHETIC DATA
    # =========================

    samples_per_digit = 1000
    latent_dim = 8
    num_runs = 5

    gen_path = f"vae_generated_ld{latent_dim}_{num_runs}runs.npy"

    cached = load_cache(gen_path)

    if cached is not None:

        generated_images = cached["x"]
        generated_labels = cached["y"]

    else:

        print("[INFO] Generating synthetic data...")

        generated_images = []
        generated_labels = []

        # =========================================
        # 5 DIFFERENT RUNS
        # =========================================
        
        for run in range(num_runs):
            
            encoder_path = os.path.join(
                CACHE_DIR,
                f"vae_encoder_run_{run+1}.keras"
            )

            decoder_path = os.path.join(
                CACHE_DIR,
                f"vae_decoder_run_{run+1}.keras"
            )


            # ---------------------------------
            # LOAD OR TRAIN VAE
            # ---------------------------------
            if os.path.exists(encoder_path) and os.path.exists(decoder_path):

                print(f"[INFO] Loading cached VAE Run {run+1}")

                encoder = tf.keras.models.load_model(
                    encoder_path
                )

                decoder = tf.keras.models.load_model(
                    decoder_path
                )

            else:

                print(f"[INFO] Training VAE Run {run+1}")

                encoder = build_encoder(latent_dim)
                decoder = build_decoder(latent_dim)

                vae = VAE(encoder, decoder)

                vae.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-4)
                )

                vae_callback = VAEVisualizationCallback(
                    decoder,
                    latent_dim=latent_dim,
                    interval=25,
                    run_id=run+1
                )

                vae.fit(
                    [x_vae, y_vae_onehot],
                    x_vae,
                    epochs=100,
                    batch_size=128,
                    verbose=1,
                    callbacks=[vae_callback]
                )

                encoder.save(encoder_path)
                decoder.save(decoder_path)

                print(f"[INFO] Saved VAE Run {run+1}")

            # ---------------------------------
            # TEMP STORAGE FOR CURRENT RUN
            # ---------------------------------
            run_images = []
            run_labels = []

            # ---------------------------------
            # GENERATE 1000/digit
            # ---------------------------------
            for digit in range(10):

                labels = np.full(samples_per_digit, digit)

                labels_onehot = tf.keras.utils.to_categorical(
                    labels,
                    10
                )

                z = np.random.normal(
                    0,
                    1,
                    (samples_per_digit, latent_dim)
                )

                imgs = decoder.predict(
                    [z, labels_onehot],
                    verbose=0
                )

                run_images.append(imgs)
                run_labels.append(labels)

            # ---------------------------------
            # CONCAT CURRENT RUN
            # ---------------------------------
            run_images = np.concatenate(run_images)
            run_labels = np.concatenate(run_labels)

            # ---------------------------------
            # SAVE VISUALIZATION OF THIS RUN
            # ---------------------------------
            save_images_grid(
                run_images[:10],
                run_labels[:10],
                filename=f"vae_run_{run+1}.png",
                n=10
            )

            print(f"[INFO] Saved VAE run {run+1} visualization")

            # ---------------------------------
            # ADD TO FINAL DATASET
            # ---------------------------------
            generated_images.append(run_images)
            generated_labels.append(run_labels)
        
        # =========================================
        # FINAL CONCATENATION
        # =========================================
        generated_images = np.concatenate(
            generated_images
        )

        generated_labels = np.concatenate(
            generated_labels
        )

        save_cache(
            gen_path,
            {
                "x": generated_images,
                "y": generated_labels
            }
        )

    print(f"[INFO] Generated dataset: {generated_images.shape}")
  
    
    # =========================
    # TRAIN / LOAD 350-ONLY CLASSIFIER
    # =========================

    classifier_path = os.path.join(
        CACHE_DIR,
        "lenet_350_only.keras"
    )

    if os.path.exists(classifier_path):

        print("[INFO] Loading cached 350-only classifier...")

        classifier = tf.keras.models.load_model(
            classifier_path
        )

    else:

        print("[INFO] Training 350-only classifier...")

        classifier = build_lenet()

        classifier.fit(
            x_small,
            y_small,
            epochs=30,
            batch_size=64,
            verbose=1
        )

        classifier.save(classifier_path)

        print("[INFO] Saved 350-only classifier")
        
    # =========================
    # CLASSIFY GENERATED DATA
    # =========================

    preds = classifier.predict(
        generated_images,
        verbose=0
    )

    confidence = np.max(preds, axis=1)

    predicted_labels = np.argmax(preds, axis=1)

    print(f"[INFO] Confidence stats:")
    print(f"Min  = {confidence.min():.3f}")
    print(f"Max  = {confidence.max():.3f}")
    print(f"Mean = {confidence.mean():.3f}")    
    
    # =========================
    # CREATE SYNTHETIC SETS
    # =========================

    # ---------- Set A ----------
    x_setA = generated_images
    y_setA = generated_labels

    # ---------- Set B ----------
    idx_B = (
        (confidence >= 0.9) &
        (predicted_labels == generated_labels)
    )

    x_setB = generated_images[idx_B]
    y_setB = generated_labels[idx_B]

    # ---------- Set C ----------
    idx_C = (
        (confidence >= 0.6) &
        (confidence < 0.9) &
        (predicted_labels == generated_labels)
    )

    x_setC = generated_images[idx_C]
    y_setC = generated_labels[idx_C]

    print(f"[INFO] Set A: {len(x_setA)}")
    print(f"[INFO] Set B: {len(x_setB)}")
    print(f"[INFO] Set C: {len(x_setC)}")
    
    # =========================
    # VISUALIZE SYNTHETIC SETS
    # =========================
    
    save_images_grid(
        x_setA[:10],
        y_setA[:10],
        filename="vae_setA.png",
        n=10
    )

    save_images_grid(
        x_setB[:10],
        y_setB[:10],
        filename="vae_setB.png",
        n=10
    )

    save_images_grid(
        x_setC[:10],
        y_setC[:10],
        filename="vae_setC.png",
        n=10
    )
    
    # =========================
    # CLASSIFIER EXPERIMENTS
    # =========================

    results = []

    # ----- Baseline 350 -----
    baseline_model = build_lenet()

    baseline_model.fit(
        x_small,
        y_small,
        epochs=30,
        batch_size=64,
        verbose=1
    )

    _, baseline_acc = baseline_model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"[BASELINE 350] {baseline_acc*100:.2f}%")

    results.append(["350 Real", baseline_acc])

    # ----- Set A -----
    acc_A = train_classifier_with_generated(
        x_small,
        y_small,
        x_setA,
        y_setA,
        x_test,
        y_test,
        "Set A"
    )

    results.append(["Set A", acc_A])

    # ----- Set B -----
    acc_B = train_classifier_with_generated(
        x_small,
        y_small,
        x_setB,
        y_setB,
        x_test,
        y_test,
        "Set B"
    )

    results.append(["Set B", acc_B])

    # ----- Set C -----
    acc_C = train_classifier_with_generated(
        x_small,
        y_small,
        x_setC,
        y_setC,
        x_test,
        y_test,
        "Set C"
    )

    results.append(["Set C", acc_C])
    
    # =========================
    # BASELINE 1000 REAL
    # =========================

    x_1000, y_1000 = get_reduced_dataset(
        x_train,
        y_train,
        1000,
        cache_name="real_1000.npy"
    )

    model_1000 = build_lenet()

    model_1000.fit(
        x_1000,
        y_1000,
        epochs=30,
        batch_size=64,
        verbose=1
    )

    _, acc_1000 = model_1000.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"[BASELINE 1000] {acc_1000*100:.2f}%")

    results.append(["1000 Real", acc_1000])
        
    print("\n===== FINAL RESULTS =====")

    for name, acc in results:
        print(f"{name}: {acc*100:.2f}%")
    
    df = pd.DataFrame(
        results,
        columns=["Experiment", "Accuracy"]
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "vae_results.csv"
    )

    df.to_csv(csv_path, index=False)

    print(f"[INFO] Saved results CSV: {csv_path}")    
    
    print("===== DONE =====")
    
# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()