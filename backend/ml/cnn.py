from __future__ import annotations

import base64
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def default_cnn_dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "chest_xray"


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def _assert_dataset_layout(dataset_root: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")


def _to_base64_png() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()
    return encoded


def _load_datasets(dataset_root: Path) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=42,
    )
    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    class_names = list(train_raw.class_names)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="train_augmentation",
    )

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    def prep_train(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        images = tf.cast(images, tf.float32)
        images = augmentation(images, training=True)
        images = preprocess(images)
        return images, labels

    def prep_eval(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        images = tf.cast(images, tf.float32)
        images = preprocess(images)
        return images, labels

    train_ds = train_raw.map(prep_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_raw.map(prep_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_raw.map(prep_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_cnn_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_pneumonia")
    return model, base_model


def _compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def _merge_history(
    history_phase_1: tf.keras.callbacks.History,
    history_phase_2: tf.keras.callbacks.History,
) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    all_keys = set(history_phase_1.history.keys()) | set(history_phase_2.history.keys())

    for key in all_keys:
        merged[key] = []
        merged[key].extend([float(v) for v in history_phase_1.history.get(key, [])])
        merged[key].extend([float(v) for v in history_phase_2.history.get(key, [])])

    return merged


def _training_curves_plot(history: dict[str, list[float]]) -> str:
    epochs = np.arange(1, len(history.get("loss", [])) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get("loss", []), label="Train Loss", linewidth=2)
    plt.plot(epochs, history.get("val_loss", []), label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get("auc", []), label="Train AUC", linewidth=2)
    plt.plot(epochs, history.get("val_auc", []), label="Val AUC", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("CNN AUC Curves")
    plt.legend()

    return _to_base64_png()


def _confusion_matrix_plot(cm: np.ndarray) -> str:
    plt.figure(figsize=(10,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred NORMAL", "Pred PNEUMONIA"],
        yticklabels=["True NORMAL", "True PNEUMONIA"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("CNN Confusion Matrix")
    return _to_base64_png()


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def _get_base_feature_model(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("Could not find base feature extractor layer for Grad-CAM.")


def make_gradcam_heatmap(
    preprocessed_img: np.ndarray | tf.Tensor,
    model: tf.keras.Model,
    base_model: tf.keras.Model,
) -> np.ndarray:
    conv_model = tf.keras.Model(inputs=model.inputs, outputs=base_model.output)

    base_model_index = model.layers.index(base_model)
    classifier_input = tf.keras.Input(shape=base_model.output_shape[1:])
    x = classifier_input
    for layer in model.layers[base_model_index + 1 :]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    input_tensor = tf.convert_to_tensor(preprocessed_img, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(input_tensor, training=False)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs, training=False)
        target = predictions[:, 0]

    grads = tape.gradient(target, conv_outputs)
    if grads is None:
        raise ValueError("Grad-CAM gradients could not be computed.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
    return heatmap.numpy()


def _fallback_input_gradient_heatmap(
    preprocessed_img: np.ndarray | tf.Tensor,
    model: tf.keras.Model,
) -> np.ndarray:
    input_tensor = tf.convert_to_tensor(preprocessed_img, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        pred = model(input_tensor, training=False)[:, 0]

    grads = tape.gradient(pred, input_tensor)
    if grads is None:
        return np.zeros(IMAGE_SIZE, dtype=np.float32)

    grad_map = np.max(np.abs(grads.numpy()[0]), axis=-1)
    max_value = float(np.max(grad_map))
    return grad_map / max_value if max_value > 0 else grad_map


def generate_gradcam_overlay(
    model: tf.keras.Model,
    img_array: np.ndarray,
    original_img: np.ndarray,
    pred_class: int,
) -> str:
    """
    Generate Grad-CAM heatmap overlaid on original image.
    Returns a base64-encoded PNG string.
    """
    import cv2

    base_model = _get_base_feature_model(model)
    conv_model = tf.keras.Model(inputs=model.inputs, outputs=base_model.output)

    base_model_index = model.layers.index(base_model)
    classifier_input = tf.keras.Input(shape=base_model.output_shape[1:])
    x = classifier_input
    for layer in model.layers[base_model_index + 1 :]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(input_tensor, training=False)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs, training=False)

        if predictions.shape[-1] == 1:
            target = predictions[:, 0] if int(pred_class) == 1 else (1.0 - predictions[:, 0])
        else:
            target = predictions[:, int(pred_class)]

    grads = tape.gradient(target, conv_outputs)
    if grads is None:
        heatmap = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.float32)
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)
        heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
        heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0)), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    original_rgb = original_img.astype(np.float32)
    overlay = (original_rgb * 0.6) + (heatmap_colored.astype(np.float32) * 0.4)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(overlay)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _collect_gradcam_samples(test_dir: Path) -> list[tuple[str, Path]]:
    classes = ["NORMAL", "PNEUMONIA"]
    extensions = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.webp"]

    selected: list[tuple[str, Path]] = []
    for class_name in classes:
        class_dir = test_dir / class_name
        paths: list[Path] = []
        for ext in extensions:
            paths.extend(sorted(class_dir.glob(ext)))
        for path in paths[:2]:
            selected.append((class_name, path))

    if len(selected) < 4:
        raise ValueError("Not enough test images to generate 4 Grad-CAM samples.")

    return selected[:4]


def _gradcam_plot(model: tf.keras.Model, dataset_root: Path) -> str:
    test_dir = dataset_root / "test"
    samples = _collect_gradcam_samples(test_dir)
    base_model = _get_base_feature_model(model)

    plt.figure(figsize=(12, 10))

    for idx, (actual_label, image_path) in enumerate(samples, start=1):
        image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)

        model_input = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(image_array.copy(), axis=0)
        )

        pred_prob = float(model.predict(model_input, verbose=0)[0][0])
        pred_label = "PNEUMONIA" if pred_prob >= 0.5 else "NORMAL"
        confidence = pred_prob if pred_label == "PNEUMONIA" else (1.0 - pred_prob)

        try:
            heatmap = make_gradcam_heatmap(
                model_input,
                model,
                base_model,
            )
        except Exception:
            heatmap = _fallback_input_gradient_heatmap(model_input, model)

        heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], IMAGE_SIZE).numpy().squeeze()

        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
        colormap = plt.colormaps.get_cmap("jet")
        colored = colormap(heatmap_uint8)[:, :, :3]

        overlay = np.clip((image_array / 255.0) * 0.6 + colored * 0.4, 0.0, 1.0)

        plt.subplot(2, 2, idx)
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(
            f"A:{actual_label} P:{pred_label} ({confidence:.2f})",
            fontsize=10,
        )

    return _to_base64_png()


@lru_cache(maxsize=1)
def _load_saved_model(model_path_str: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path_str)


def train_and_evaluate_cnn(
    dataset_root: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root) if dataset_root is not None else default_cnn_dataset_root()
    models_dir = Path(models_dir) if models_dir is not None else _default_models_dir()

    _assert_dataset_layout(dataset_root)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = _load_datasets(dataset_root)

    model, base_model = build_cnn_model()

    _compile_model(model, learning_rate=0.001)
    history_phase_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=2,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    _compile_model(model, learning_rate=1e-5)
    history_phase_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=10,
        epochs=20,
        verbose=2,
    )

    model_path = models_dir / "cnn_model.h5"
    model.save(model_path)
    _load_saved_model.cache_clear()

    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_true = np.concatenate([labels.numpy().ravel() for _, labels in test_ds]).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": _safe_auc(y_true, y_prob),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    merged_history = _merge_history(history_phase_1, history_phase_2)
    training_curves_b64 = _training_curves_plot(merged_history)
    confusion_matrix_b64 = _confusion_matrix_plot(cm)
    gradcam_b64 = _gradcam_plot(model, dataset_root)

    results: dict[str, Any] = {
        "task": "cnn_pneumonia_classification",
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "metrics": metrics,
        "training_curves_plot": training_curves_b64,
        "gradcam_plot": gradcam_b64,
        "confusion_matrix_plot": confusion_matrix_b64,
        "model_path": str(model_path),
    }

    with (models_dir / "cnn_results.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp)

    return results


def predict_cnn_image(
    image_bytes: bytes,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    models_dir = Path(models_dir) if models_dir is not None else _default_models_dir()
    model_path = models_dir / "cnn_model.h5"

    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model not found at {model_path}. Train first via /dl/cnn/train."
        )

    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)

    model_input = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(image_array, axis=0)
    )

    model = _load_saved_model(str(model_path))
    prob = float(model.predict(model_input, verbose=0)[0][0])

    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else (1.0 - prob)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
    }
