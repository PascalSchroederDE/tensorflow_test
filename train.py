import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
BATCH_SIZE = 32
BUFFER_SIZE = 10000

CSV_LOCATION = 'data/train.csv'
FILE_DIRECTORY = 'train/cut/'
ID_COLUMN = 'id'
LABEL_COLUMN = 'has_scratch'

TRAIN_SIZE = 0.8

LEARNING_RATE = 0.0001

NUM_EPOCHS = 30
VAL_STEPS = 10


def read_csv(csv_location):
    train_csv = pd.read_csv(csv_location)
    filenames = [FILE_DIRECTORY + fname for fname in train_csv[ID_COLUMN].tolist()]
    labels = train_csv[LABEL_COLUMN].tolist()

    return filenames, labels


def build_dataset(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
    dataset = (dataset.map(parse_filename).shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE))
    return dataset


def parse_filename(filename, label=None, img_width = IMAGE_WIDTH, img_height = IMAGE_HEIGHT):
    img_string = tf.io.read_file(filename)
    img_decoded = tf.image.decode_png(img_string)
    img_norm = (tf.cast(img_decoded, tf.float32)/127.5)-1
    img_resized = tf.image.resize(img_norm, (img_width, img_height))
    return img_resized, label


def build_model():
    base_model = keras.applications.MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    maxpool_layer = keras.layers.GlobalMaxPooling2D()
    prediction_layer = keras.layers.Dense(1, activation='sigmoid')

    model = keras.Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])
    learning_rate = LEARNING_RATE

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), 
                loss='binary_crossentropy',
                metrics=['accuracy']
    )

    return model


def train_model(model, train_data, num_train, val_data, num_epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, val_steps = VAL_STEPS):
    steps_per_epoch = round(num_train)//batch_size
    history = model.fit(train_data.repeat(), epochs=num_epochs, steps_per_epoch = steps_per_epoch, validation_data = val_data.repeat(), validation_steps = VAL_STEPS)
    return history


def main():
    filenames, labels = read_csv(CSV_LOCATION)
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames, labels, train_size=TRAIN_SIZE, random_state=42)

    train_data = build_dataset(train_filenames, train_labels)
    val_data = build_dataset(val_filenames, val_labels)

    model = build_model()
    history = train_model(model, train_data, len(train_filenames), val_data)
    model.save('model/weights_epoch_30.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print(acc, val_acc, loss, val_loss)


if __name__ == '__main__':
    main()