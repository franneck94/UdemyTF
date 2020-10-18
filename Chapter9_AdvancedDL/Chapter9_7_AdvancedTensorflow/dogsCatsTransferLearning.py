import tensorflow as tf

from dogsCatsData import IMG_SHAPE
from dogsCatsData import NUM_OUTPUTS
from dogsCatsData import get_dataset


def model_fn() -> tf.keras.Model:
    """Build the transfer learning model.

    Returns
    -------
    tf.keras.Model
        The extended movile net model
    """
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=IMG_SHAPE,
        classes=NUM_OUTPUTS,
    )
    # base_model.summary()
    # base_model.trainable = False
    print(f"Number of layers in the base model: {len(base_model.layers)}")
    fine_tune_at = len(base_model.layers) - 10
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=NUM_OUTPUTS)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_model(model: tf.keras.Model) -> None:
    """Train and test the transfer learning model

    Parameters
    ----------
    model : tf.keras.Model
        The transfer learning model
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metric = tf.keras.metrics.BinaryAccuracy()
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
    # Own model had accuracy of 0.925 on test set
    model = model_fn()
    train_model(model)
