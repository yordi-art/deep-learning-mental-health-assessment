"""
model.py
Defines the ANN architecture for mental health severity classification.

Architecture:
  Input (16) → Dense(128) → BN → Dropout(0.3)
             → Dense(64)  → BN → Dropout(0.3)
             → Dense(32)  → BN → Dropout(0.2)
             → Output(4, softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_model(input_dim: int = 16, num_classes: int = 4, learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Build and compile the ANN classifier.

    - BatchNormalization stabilizes training.
    - Dropout prevents overfitting on small datasets.
    - L2 regularization on Dense layers adds weight penalty.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="questionnaire_input")

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="severity_output")(x)

    model = tf.keras.Model(inputs, outputs, name="MentalHealthANN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
