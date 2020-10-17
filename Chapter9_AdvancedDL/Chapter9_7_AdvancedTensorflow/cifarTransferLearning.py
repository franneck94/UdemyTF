import tensorflow as tf

from cifarData import get_dataset


IMG_SIZE = 32
IMG_DEPTH = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)
NUM_CLASSES = 10


def model_fn() -> tf.keras.Model:
    """Build the transfer learning model.

    Returns
    -------
    tf.keras.Model
        The extended movile net model
    """
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        input_shape=IMG_SHAPE,
        include_top=False,
        classes=NUM_CLASSES,
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=NUM_CLASSES)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_model(model: tf.keras.Model) -> None:
    """Train and test the transfer learning model

    Parameters
    ----------
    model : tf.keras.Model
        The transfer learning model
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.CategoricalAccuracy()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metric],
    )
    model.summary()
    train_dataset, validation_dataset, test_dataset = get_dataset()
    model.fit(train_dataset, epochs=20, validation_data=validation_dataset)
    model.evaluate(test_dataset)


if __name__ == "__main__":
    model = model_fn()
    train_model(model)
