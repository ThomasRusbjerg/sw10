import sys
import datetime
import argparse
import hypertune
import tensorflow as tf
import numpy as np
from tensorflow import keras
from datetime import date, datetime

import gs_utils

def _parse_args(argv):
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-dir',
        help='Output directory for exporting model and other metadata.',
        required=False,
        type=str,
        default="gs://sw10-bucket/omr/jobs/" + date.today().strftime("%Y%m%d") + "-" + datetime.now().strftime("%H%M%S") + "/1"
    )

    parser.add_argument(
        '--epochs',
        help='Epochs',
        required=False,
        type=int,
        default=1
    )

    parser.add_argument(
        '--batch_size',
        help='Batch size',
        required=False,
        type=int,
        default=15
    )

    return parser.parse_args(argv)

def dummy_model(x_train, y_train, input_shape, num_classes, flags):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=flags.job_dir + "/logs")]

    history = model.fit(x_train, y_train, batch_size=flags.batch_size, epochs=flags.epochs, validation_split=0.1, callbacks=callbacks)
    return model, history


def report_hyperparam(name, metric):
    """ report metric to AI platform hyperparameter tuner """
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(\
        hyperparameter_metric_tag=name,\
        metric_value=metric,\
        global_step=1)

def main():
    """ main function """
    # parse flags
    flags = _parse_args(sys.argv[1:])
    print("Running with configurations: " + str(flags) + "\n")

    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model, history = dummy_model(x_train, y_train, input_shape, num_classes, flags)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    # save the model
    modelname = 'model.h5'
    model.save(modelname)
    if flags.job_dir:
        gs_utils.upload_blob(flags.job_dir, modelname)

    # report the minimum val_loss metric we get when training
    loss = history.history['val_loss']
    report_hyperparam('val_loss', min(loss))

if __name__ == "__main__":
    main()
