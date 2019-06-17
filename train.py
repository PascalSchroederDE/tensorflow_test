import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import logging

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


def read_csv(mount, csv_location, file_prefix=FILE_DIRECTORY, input_col=ID_COLUMN, target=LABEL_COLUMN):
    train_csv = pd.read_csv(mount + csv_location)
    input = [mount + file_prefix + "/" + fname for fname in train_csv[input_col].tolist()]
    labels = train_csv[target].tolist()

    return input, labels


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


def build_model(learning_rate = LEARNING_RATE):
    base_model = keras.applications.MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    maxpool_layer = keras.layers.GlobalMaxPooling2D()
    prediction_layer = keras.layers.Dense(1, activation='sigmoid')

    model = keras.Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])
    learning_rate = learning_rate

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), 
                loss='binary_crossentropy',
                metrics=['accuracy']
    )

    return model


def train_model(model, train_data, num_train, val_data, num_epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, val_steps = VAL_STEPS):
    steps_per_epoch = int(round(num_train)//batch_size)
    history = model.fit(train_data.repeat(), epochs=num_epochs, steps_per_epoch = steps_per_epoch, validation_data = val_data.repeat(), validation_steps = val_steps)
    return history


def main():
    parser = argparse.ArgumentParser(description="Image Trainer")
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--validations', type=int, help='Number of validation stepts')
    parser.add_argument('--workers' , type=int, help='Number of workers to use for training')
    parser.add_argument('--pvc_path' , type=str, help='Mount path')
    parser.add_argument('--trainset', type=str, help='Path of training set')
    parser.add_argument('--input', type=str, help='Path of csv with labels for each file of training set')
    parser.add_argument('--filenames', type=str, help='Input column of csv')
    parser.add_argument('--target', type=str, help='Target column name of csv')
    parser.add_argument('--train_size', type=float, help='Size of training set')
    parser.add_argument('--learn_rate', type=float, help='Learning rate for model training')
    parser.add_argument('--output', type=str, help='path of calculated model')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info('Reading CSV..')
    filenames, labels = read_csv(mount=args.pvc_path, csv_location=args.input, file_prefix=args.trainset, input_col=args.filenames, target=args.target)
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames, labels, train_size=args.train_size, random_state=42)

    logging.info('Building Dataset...')
    train_data = build_dataset(train_filenames, train_labels)
    val_data = build_dataset(val_filenames, val_labels)

    logging.info('Building model...')
    model = build_model(args.learn_rate)
    logging.info('Training model...')
    history = train_model(model, train_data, len(train_filenames), val_data, num_epochs=args.epochs, val_steps=args.validations)
    logging.info('Saving model...')
    print(args.output)
    model.save(args.output)

    with open('/output.txt', 'w') as f:
      f.write(args.output)


if __name__ == '__main__':
    main()