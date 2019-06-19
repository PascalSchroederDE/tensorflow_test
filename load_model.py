import tensorflow as tf
from tensorflow import keras
import pandas as pd
import argparse
import logging

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
BATCH_SIZE = 32
BUFFER_SIZE = 10000

CSV_LOCATION = 'data/train.csv'
MODEL_LOCATION = 'model/weights_epoch_30.h5'
FILE_DIRECTORY = 'train/cut/'
ID_COLUMN = 'id'
LABEL_COLUMN = 'has_scratch'


def read_csv(mount, csv_location, directory = FILE_DIRECTORY, input_col = ID_COLUMN, target=LABEL_COLUMN):
    train_csv = pd.read_csv(mount + csv_location)
    filenames = [mount + directory + "/" + fname for fname in train_csv[input_col].tolist()]
    labels = train_csv[target].tolist()

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
    parser = argparse.ArgumentParser(description="Image Trainer")
    parser.add_argument('--workers' , type=int, help='Number of workers to use for training')
    parser.add_argument('--pvc_path' , type=str, help='Mount path')
    parser.add_argument('--testset', type=str, help='Path of training set')
    parser.add_argument('--input', type=str, help='Path of csv with labels for each file of training set')
    parser.add_argument('--filenames', type=str, help='Input column of csv')
    parser.add_argument('--target', type=str, help='Target column name of csv')
    parser.add_argument('--model', type=str, help='Path of given model')
    parser.add_argument('--output', type=str, help='path of results')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Loading model...")
    model = keras.models.load_model(args.model)
    #model.summary()

    logging.info("Reading csv...")
    filenames, labels = read_csv(mount=args.pvc_path, csv_location=args.input, directory=args.testset, input_col=args.filenames, target=args.target)

    logging.info("Loading dataset...")
    data = build_dataset(filenames, labels)    
    loss, acc = model.evaluate(data, steps=1)

    logging.info("Writing results to csv")
    output_file = open(args.output, 'w')
    output_file.write("Loss: {}, Acc: {}".format(loss, acc))
    output_file.close()

if __name__ == '__main__':
    main()