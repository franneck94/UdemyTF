import tensorflow as tf

from cifarData import get_dataset

IMG_SIZE = 32
IMG_DEPTH = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)
NUM_CLASSES = 10


def model_fn():
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
    outputs = tf.keras.layers.Dense(NUM_CLASSES)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


model = model_fn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)
model.summary()

train_dataset, validation_dataset, test_dataset = get_dataset()

epochs = 20
model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)
