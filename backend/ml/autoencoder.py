from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = (128, 128)
IMAGE_SHAPE = (128, 128, 1)
NOISE_FACTOR = 0.3
EPOCHS = 50
BATCH_SIZE = 32


def default_autoencoder_dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "chest_xray"


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def _to_base64_png() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()
    return encoded


def _collect_split_image_paths(split_dir: Path) -> list[Path]:
    extensions = ("*.jpeg", "*.jpg", "*.png", "*.bmp", "*.webp")
    image_paths: list[Path] = []

    for ext in extensions:
        image_paths.extend(sorted(split_dir.rglob(ext)))

    return image_paths


def _assert_dataset_layout(dataset_root: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

        if not _collect_split_image_paths(split_dir):
            raise FileNotFoundError(
                f"No image files found under {split_dir}. Expected class folders with X-ray images."
            )


def _load_split_images(split_dir: Path) -> np.ndarray:
    image_paths = _collect_split_image_paths(split_dir)
    if not image_paths:
        raise ValueError(f"No images found for split directory: {split_dir}")

    images: list[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize(IMAGE_SIZE)
            arr = np.asarray(gray, dtype=np.float32) / 255.0
            images.append(arr[..., np.newaxis])

    return np.stack(images, axis=0).astype(np.float32)


def add_gaussian_noise(clean_images: np.ndarray, noise_factor: float = NOISE_FACTOR) -> np.ndarray:
    noisy = clean_images + noise_factor * np.random.normal(size=clean_images.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.astype(np.float32)


def build_autoencoder(input_shape: tuple[int, int, int] = IMAGE_SHAPE) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    bottleneck = tf.keras.layers.MaxPooling2D(2, name="bottleneck")(x)

    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(bottleneck)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    outputs = tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="xray_denoising_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


def _loss_curve_plot(history: tf.keras.callbacks.History) -> str:
    history_data = history.history
    epochs = np.arange(1, len(history_data.get("loss", [])) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history_data.get("loss", []), label="Train Loss", linewidth=2)
    plt.plot(epochs, history_data.get("val_loss", []), label="Val Loss", linewidth=2)
    plt.title("Autoencoder Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    return _to_base64_png()


def _comparison_plot(
    noisy_images: np.ndarray,
    reconstructed_images: np.ndarray,
    clean_images: np.ndarray,
) -> str:
    n_images = int(min(5, clean_images.shape[0]))
    if n_images <= 0:
        raise ValueError("Need at least one test image to render comparison plots.")

    plt.figure(figsize=(9, 3 * n_images))
    for idx in range(n_images):
        plt.subplot(n_images, 3, idx * 3 + 1)
        plt.imshow(noisy_images[idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")
        if idx == 0:
            plt.title("Noisy")

        plt.subplot(n_images, 3, idx * 3 + 2)
        plt.imshow(reconstructed_images[idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")
        if idx == 0:
            plt.title("Reconstructed")

        plt.subplot(n_images, 3, idx * 3 + 3)
        plt.imshow(clean_images[idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")
        if idx == 0:
            plt.title("Original")

    return _to_base64_png()


def _single_comparison_image(noisy: np.ndarray, reconstructed: np.ndarray, clean: np.ndarray) -> str:
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(noisy.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Noisy")

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Reconstructed")

    plt.subplot(1, 3, 3)
    plt.imshow(clean.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Original")

    return _to_base64_png()


def _comparison_images(
    noisy_images: np.ndarray,
    reconstructed_images: np.ndarray,
    clean_images: np.ndarray,
) -> list[str]:
    n_images = int(min(5, clean_images.shape[0]))
    return [
        _single_comparison_image(noisy_images[i], reconstructed_images[i], clean_images[i])
        for i in range(n_images)
    ]


def train_and_evaluate_autoencoder(
    dataset_root: str | Path | None = None,
    models_dir: str | Path | None = None,
    noise_factor: float = NOISE_FACTOR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root) if dataset_root is not None else default_autoencoder_dataset_root()
    models_dir = Path(models_dir) if models_dir is not None else _default_models_dir()

    _assert_dataset_layout(dataset_root)
    models_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    train_clean = _load_split_images(dataset_root / "train")
    val_clean = _load_split_images(dataset_root / "val")
    test_clean = _load_split_images(dataset_root / "test")

    train_noisy = add_gaussian_noise(train_clean, noise_factor=noise_factor)
    val_noisy = add_gaussian_noise(val_clean, noise_factor=noise_factor)
    test_noisy = add_gaussian_noise(test_clean, noise_factor=noise_factor)

    model = build_autoencoder(input_shape=IMAGE_SHAPE)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
        )
    ]

    history = model.fit(
        train_noisy,
        train_clean,
        validation_data=(val_noisy, val_clean),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    model_path = models_dir / "autoencoder.h5"
    model.save(model_path)

    evaluation = model.evaluate(test_noisy, test_clean, return_dict=True, verbose=0)
    test_mse = float(evaluation.get("mse", evaluation.get("loss", np.nan)))

    n_samples = int(min(5, test_clean.shape[0]))
    sample_noisy = test_noisy[:n_samples]
    sample_clean = test_clean[:n_samples]
    sample_reconstructed = model.predict(sample_noisy, verbose=0)

    comparison_plot_b64 = _comparison_plot(sample_noisy, sample_reconstructed, sample_clean)
    comparison_images_b64 = _comparison_images(sample_noisy, sample_reconstructed, sample_clean)
    loss_curve_b64 = _loss_curve_plot(history)

    results: dict[str, Any] = {
        "task": "xray_denoising_autoencoder",
        "dataset_root": str(dataset_root),
        "image_shape": list(IMAGE_SHAPE),
        "noise_factor": float(noise_factor),
        "metrics": {
            "test_mse": test_mse,
        },
        "loss_curve_plot": loss_curve_b64,
        "comparison_plot": comparison_plot_b64,
        "comparison_images": comparison_images_b64,
        "model_path": str(model_path),
    }

    with (models_dir / "autoencoder_results.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp)

    return results


if __name__ == "__main__":
    output = train_and_evaluate_autoencoder()
    print(json.dumps(output.get("metrics", {}), indent=2))
