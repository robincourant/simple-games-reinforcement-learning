import os
import pickle
from typing import List

from keras.models import Sequential
from keras.layers import Dense, Dropout


def create_mlp_model(
    n_features: int,
    n_categories: int,
    batch_size: int = None,
    loss: str = "categorical_crossentropy",
    optimizer: str = "adam",
    proba_output: bool = True,
    metrics: List[str] = ["accuracy"],
) -> Sequential:
    """Build and compile a MLP classifier with 6 layers and dropout.

    :param n_features: number of input features (input shape).
    :param n_categories: number of output categories (output shape).
    :param batch_size: [description]
    :param loss: [description]
    :param optimizer: [description]
    :param proba_output: [description]
    :param metrics: [description]
    :return: compiled model.
    """
    model = Sequential()

    model.add(Dense(128, input_shape=(n_features, ), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    if proba_output:
        model.add(Dense(n_categories, activation="softmax"))
    else:
        model.add(Dense(n_categories))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def save_model(
    model: Sequential,
    file_path: str,
):
    """Save a model in `pickle` format.

    :param model: model to be saved.
    :param file_path: filepath or folderpath.
    """
    folder_path = os.path.dirname(file_path)
    # Check output folder exists, or create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the model
    with open(file_path, "wb") as dump_file:
        pickle.dump(model, dump_file)
    print("Model saved.")
