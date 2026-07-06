"""
Brain Tumor Detection - Model Training Script
------------------------------------------------
Trains a Convolutional Neural Network (CNN) to classify brain MRI images
as Tumor / No Tumor.

Dataset expected structure (download from Kaggle first - see README.md):

    brain_tumor_dataset/
        yes/   -> MRI images WITH a tumor
        no/    -> MRI images WITHOUT a tumor

Run:
    python train.py

Output:
    - Prints training/validation accuracy per epoch
    - Prints a final classification report + confusion matrix
    - Saves the trained model as brain_tumor_model.h5
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = "brain_tumor_dataset"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25


def build_model():
    """A CNN built from scratch: 3 conv blocks + dense classifier head."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),  # binary output: tumor vs no tumor
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    print("Class indices (0/1 mapping):", train_gen.class_indices)

    model = build_model()
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop],
    )

    # Evaluate on the validation set
    val_gen.reset()
    predictions = (model.predict(val_gen) > 0.5).astype("int32").flatten()
    true_labels = val_gen.classes
    class_names = list(train_gen.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predictions))

    best_val_acc = max(history.history["val_accuracy"]) * 100
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print("^ This is YOUR real number. Use it on the resume, not a guess.")

    model.save("brain_tumor_model.h5")
    print("Model saved as brain_tumor_model.h5")


if __name__ == "__main__":
    main()
