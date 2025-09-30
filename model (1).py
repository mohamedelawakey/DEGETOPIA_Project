
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape=(224, 224, 3), num_classes=1):
    base = keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=input_shape,
        weights=None
    )
    base.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)

    model = keras.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    return model
