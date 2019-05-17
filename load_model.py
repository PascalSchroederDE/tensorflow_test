import tensorflow as tf
from tensorflow import keras
import pandas as pd

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
BATCH_SIZE = 32
BUFFER_SIZE = 10000

CSV_LOCATION = 'data/train.csv'
MODEL_LOCATION = 'model/weights_epoch_30.h5'
FILE_DIRECTORY = 'train/cut/'
ID_COLUMN = 'id'
LABEL_COLUMN = 'has_scratch'


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


def main():
    model = keras.models.load_model(MODEL_LOCATION)
    model.summary()

    filenames, labels = read_csv(CSV_LOCATION)

    data = build_dataset(filenames, labels)    
    loss, acc = model.evaluate(data)

    print(100*acc)

if __name__ == '__main__':
    main()