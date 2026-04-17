"""
trainer.py
Handles model training with early stopping, learning rate scheduling,
and model checkpointing.
"""

import tensorflow as tf


def train(
    model: tf.keras.Model,
    data: dict,
    epochs: int = 100,
    batch_size: int = 32,
    model_path: str = "models/mental_health_model.keras",
) -> tf.keras.callbacks.History:
    """
    Train the model with:
    - EarlyStopping    : stops when val_loss stops improving (patience=15)
    - ReduceLROnPlateau: halves LR when val_loss plateaus (patience=7)
    - ModelCheckpoint  : saves the best model weights automatically
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nBest model saved → {model_path}")
    return history
