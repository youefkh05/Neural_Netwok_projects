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
 
    # KEEP in [0,1] for GAN
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
    plt.title("GAN Synthetic Data Results")

    plt.xticks(rotation=20)

    path = os.path.join(
        OUTPUT_DIR,
        "GAN_results_plot.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved plot: {path}")

# =========================
# BUILD CONDITIONAL GENERATOR MODEL
# =========================
def build_generator(latent_dim=100):

    z_input = tf.keras.Input(shape=(latent_dim,))
    label_input = tf.keras.Input(shape=(10,))

    x = tf.keras.layers.Concatenate()(
        [z_input, label_input]
    )

    x = tf.keras.layers.Dense(7*7*256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Reshape((7,7,256))(x)

    x = tf.keras.layers.Conv2DTranspose(
        128,
        4,
        strides=2,
        padding="same"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(
        64,
        4,
        strides=2,
        padding="same"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    output = tf.keras.layers.Conv2D(
        1,
        3,
        padding="same",
        activation="sigmoid"
    )(x)

    return tf.keras.Model(
        [z_input, label_input],
        output
    )
    
# =========================
# BUILD CONDITIONAL DISCRIMINATOR MODEL
# =========================
def build_discriminator():

    image_input = tf.keras.Input(
        shape=(28,28,1)
    )

    label_input = tf.keras.Input(
        shape=(10,)
    )

    label_map = tf.keras.layers.Dense(28*28)(
        label_input
    )

    label_map = tf.keras.layers.Reshape(
        (28,28,1)
    )(label_map)

    x = tf.keras.layers.Concatenate()(
        [image_input, label_map]
    )

    x = tf.keras.layers.Conv2D(
        64,
        3,
        strides=2,
        padding="same"
    )(x)

    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(
        128,
        3,
        strides=2,
        padding="same"
    )(x)

    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    return tf.keras.Model(
        [image_input, label_input],
        output
    )

    
# =========================
# BUILD COMPLETE cGAN
# =========================
def build_gan(generator, discriminator, latent_dim=100):

    # Freeze discriminator during GAN training
    discriminator.trainable = False

    # Inputs
    noise_input = tf.keras.Input(shape=(latent_dim,))
    label_input = tf.keras.Input(shape=(10,))

    # Generate fake image
    fake_img = generator([noise_input, label_input])

    # Discriminator decision
    validity = discriminator([fake_img, label_input])

    # Combined GAN model
    gan = tf.keras.Model(
        [noise_input, label_input],
        validity
    )

    gan.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002,
            beta_1=0.5
        ),
        loss="binary_crossentropy"
    )

    return gan


# =========================
# TRAIN CONDITIONAL GAN
# =========================
def train_cgan(
    generator,
    discriminator,
    gan,
    x_train,
    y_train,
    epochs=100,
    batch_size=128,
    latent_dim=100,
    run_id=1
):

    half_batch = batch_size // 2

    # One-hot labels
    y_train_onehot = tf.keras.utils.to_categorical(
        y_train,
        10
    )

    # For plotting losses
    d_losses = []
    g_losses = []

    for epoch in range(epochs):

        # =====================================
        # TRAIN DISCRIMINATOR
        # =====================================

        # -------------------------
        # REAL IMAGES
        # -------------------------
        idx = np.random.randint(
            0,
            x_train.shape[0],
            half_batch
        )

        real_imgs = x_train[idx]
        real_labels = y_train_onehot[idx]

        real_y = np.ones((half_batch, 1))

        # -------------------------
        # FAKE IMAGES
        # -------------------------
        noise = np.random.normal(
            0,
            1,
            (half_batch, latent_dim)
        )

        fake_digits = np.random.randint(
            0,
            10,
            half_batch
        )

        fake_labels = tf.keras.utils.to_categorical(
            fake_digits,
            10
        )

        fake_imgs = generator.predict(
            [noise, fake_labels],
            verbose=0
        )

        fake_y = np.zeros((half_batch, 1))

        # -------------------------
        # Train discriminator
        # -------------------------
        discriminator.trainable = True
        
        d_loss_real = discriminator.train_on_batch(
            [real_imgs, real_labels],
            real_y
        )

        d_loss_fake = discriminator.train_on_batch(
            [fake_imgs, fake_labels],
            fake_y
        )

        d_loss = 0.5 * np.add(
            d_loss_real,
            d_loss_fake
        )

        # =====================================
        # TRAIN GENERATOR
        # =====================================

        noise = np.random.normal(
            0,
            1,
            (batch_size, latent_dim)
        )

        sampled_digits = np.random.randint(
            0,
            10,
            batch_size
        )

        sampled_labels = tf.keras.utils.to_categorical(
            sampled_digits,
            10
        )

        valid_y = np.ones((batch_size, 1))
        
        discriminator.trainable = False

        g_loss = gan.train_on_batch(
            [noise, sampled_labels],
            valid_y
        )

        # Save losses
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        # =====================================
        # LOGGING
        # =====================================
        if epoch % 10 == 0:

            print(
                f"[Epoch {epoch}] "
                f"D Loss: {d_loss[0]:.4f} | "
                f"D Acc: {100*d_loss[1]:.2f}% | "
                f"G Loss: {g_loss:.4f}"
            )

        # =====================================
        # VISUALIZATION
        # =====================================
        if epoch % 20 == 0:

            test_noise = np.random.normal(
                0,
                1,
                (10, latent_dim)
            )

            test_digits = np.arange(10)

            test_labels = tf.keras.utils.to_categorical(
                test_digits,
                10
            )

            generated = generator.predict(
                [test_noise, test_labels],
                verbose=0
            )

            save_images_grid(
                generated,
                test_digits,
                filename=f"gan_run_{run_id}_epoch_{epoch}.png",
                n=10
            )

    # =====================================
    # LOSS CURVE
    # =====================================
    plt.figure(figsize=(8,5))

    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("GAN Training Loss")

    plt.legend()

    path = os.path.join(
        OUTPUT_DIR,
        f"gan_loss_curve_run_{run_id}.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] Saved GAN loss curve: {path}")

# =========================
# TRAIN CLASSIFIER WITH GENERATED DATA
# =========================
def train_classifier_with_generated(
    x_real,
    y_real,
    x_fake,
    y_fake,
    x_test,
    y_test,
    title="Experiment"
):

    x_final = np.concatenate([x_real, x_fake])
    y_final = np.concatenate([y_real, y_fake])

    x_final, y_final = shuffle_dataset(
        x_final,
        y_final
    )

    model = build_lenet()

    model.fit(
        x_final,
        y_final,
        epochs=30,
        batch_size=64,
        verbose=1
    )

    _, acc = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"[{title}] Accuracy: {acc*100:.2f}%")

    return acc

    
# =========================
# MAIN FUNCTION (GAN PIPELINE)
# =========================
def main():

    print("===== GAN PIPELINE START =====")

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
        cache_name=f"aug_GAN_{real_n}.npy"
    )
    
    # Visualize some augmented samples
    save_images_grid(
        x_aug[:5],
        y_aug[:5],
        aug_types=aug_types[:5],
        filename="GAN_aug_samples.png",
        n=10
    )


   # Combine real + augmented
    x_gan = np.concatenate([x_small, x_aug])
    y_gan = np.concatenate([y_small, y_aug])
    
    x_gan, y_gan = shuffle_dataset(
        x_gan,
        y_gan
    )

    print(f"[INFO] GAN training dataset: {x_gan.shape}")
    
    
    # =========================
    # TRAIN / LOAD 350-ONLY CLASSIFIER
    # =========================

    classifier_path = os.path.join(
        CACHE_DIR,
        "lenet_350_only.keras"
    )

    if os.path.exists(classifier_path):

        print("[INFO] Loading cached classifier...")

        classifier = tf.keras.models.load_model(
            classifier_path
        )

    else:

        print("[INFO] Training classifier...")

        classifier = build_lenet()

        classifier.fit(
            x_small,
            y_small,
            epochs=30,
            batch_size=64,
            verbose=1
        )

        classifier.save(classifier_path)   
        
    # =========================
    # LOAD GENERATED CACHE
    # =========================

    cached = load_cache("gan_generated.npy")

    if cached is not None:

        generated_images = cached["x"]
        generated_labels = cached["y"]

        print("[INFO] Loaded cached GAN generated dataset")

    else:

        # =========================
        # GAN GENERATION
        # =========================

        samples_per_digit = 1000
        latent_dim = 100
        num_runs = 5

        generated_images = []
        generated_labels = []

        for run in range(num_runs):

            print(f"\n[INFO] GAN Run {run+1}")

            gen_path = os.path.join(    
                CACHE_DIR,
                f"cgan_generator_run_{run+1}.keras"
            )

            disc_path = os.path.join(
                CACHE_DIR,
                f"cgan_discriminator_run_{run+1}.keras"
            )

            if os.path.exists(gen_path) and os.path.exists(disc_path):

                print(f"[INFO] Loading GAN Run {run+1}")

                generator = tf.keras.models.load_model(gen_path)

                discriminator = tf.keras.models.load_model(disc_path)
                
                discriminator.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        0.0002,
                        0.5
                    ),
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )
                
                gan = build_gan(
                    generator,
                    discriminator
                )

            else:

                print(f"[INFO] Training GAN Run {run+1}")

                generator = build_generator(latent_dim)

                discriminator = build_discriminator()

                discriminator.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        0.0002,
                        0.5
                    ),
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )

                
                gan = build_gan(
                            generator,
                            discriminator
                        )
                
                train_cgan(
                    generator,
                    discriminator,
                    gan,
                    x_gan,
                    y_gan,
                    epochs=200,
                    batch_size=128,
                    latent_dim=latent_dim,
                    run_id=run+1
                )

                generator.save(gen_path)
                discriminator.save(disc_path)

            
            # -------------------------
            # Generate 1000/digit
            # -------------------------
            run_images = []
            run_labels = []

            for digit in range(10):

                labels = np.full(
                    samples_per_digit,
                    digit
                )

                labels_onehot = tf.keras.utils.to_categorical(
                    labels,
                    10
                )

                noise = np.random.normal(
                    0,
                    1,
                    (samples_per_digit, latent_dim)
                )

                imgs = generator.predict(
                    [noise, labels_onehot],
                    verbose=0
                )

                run_images.append(imgs)
                run_labels.append(labels)

            run_images = np.concatenate(run_images)
            run_labels = np.concatenate(run_labels)

            # -------------------------
            # Visualize Run
            # -------------------------
            save_images_grid(
                run_images[:10],
                run_labels[:10],
                filename=f"gan_run_{run+1}.png",
                n=10
            )

            generated_images.append(run_images)
            generated_labels.append(run_labels)   



    
        
        # =========================
        # Final Concatenation
        # =========================

        generated_images = np.concatenate(
            generated_images
        )

        generated_labels = np.concatenate(
            generated_labels
        )

        print(f"[INFO] Generated dataset: {generated_images.shape}")
        
        save_cache(
            "gan_generated.npy",
            {
                "x": generated_images,
                "y": generated_labels
            }
        )
        
    
    
    # =========================
    # CLASSIFY GENERATED DATA
    # =========================

    preds = classifier.predict(
        generated_images,
        verbose=0
    )

    confidence = np.max(preds, axis=1)

    predicted_labels = np.argmax(
        preds,
        axis=1
    )

    print(f"[INFO] Confidence Mean: {confidence.mean():.3f}")   
    
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
    
    save_images_grid(
        x_setA[:10],
        y_setA[:10],
        filename="gan_setA.png",
        n=10
    )

    save_images_grid(
        x_setB[:10],
        y_setB[:10],
        filename="gan_setB.png",
        n=10
    )

    save_images_grid(
        x_setC[:10],
        y_setC[:10],
        filename="gan_setC.png",
        n=10
    ) 

    # =========================
    # CLASSIFIER EXPERIMENTS
    # =========================

    results = []

    # -------------------------
    # Baseline 350
    # -------------------------
    
    _, baseline_acc = classifier.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    results.append(["350 Real", baseline_acc])

    print(f"[BASELINE 350] {baseline_acc*100:.2f}%")

    # -------------------------
    # Set A
    # -------------------------
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

    # -------------------------
    # Set B
    # -------------------------
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

    # -------------------------
    # Set C
    # -------------------------
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

    # -------------------------
    # Baseline 1000
    # -------------------------
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

    results.append(["1000 Real", acc_1000])

    print(f"[BASELINE 1000] {acc_1000*100:.2f}%") 

    df = pd.DataFrame(
        results,
        columns=["Experiment", "Accuracy"]
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "gan_results.csv"
    )

    df.to_csv(csv_path, index=False)

    plot_results_from_csv(csv_path)

    print(f"[INFO] Saved results CSV: {csv_path}")   
    
    print("\n===== FINAL RESULTS =====")

    for name, acc in results:
        print(f"{name}: {acc*100:.2f}%")
    
    print("===== DONE =====")
    
# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()