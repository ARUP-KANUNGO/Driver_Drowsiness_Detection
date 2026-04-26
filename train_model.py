import tensorflow as tf
import matplotlib.pyplot as plt
import time


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3

train_dir = "data/train"
val_dir = "data/val"

AUTOTUNE = tf.data.AUTOTUNE


train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)


train_data = train_data.shuffle(200)

train_data = train_data.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)
val_data = val_data.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)

# use prefetch ONLY (no RAM cache)
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)


base_model = tf.keras.applications.MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.summary()


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "model/best_model.keras",
        save_best_only=True
    )
]

start_time = time.time()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

end_time = time.time()
print("⏱ Training Time:", end_time - start_time, "seconds")


model.save("model/drowsiness_model.keras")
print("✅ Training complete. Model saved.")


plt.clf()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.savefig("accuracy.png")
plt.show()

plt.clf()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.savefig("loss.png")
plt.show()