import os
import pickle
from typing import Any, List, Tuple

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Add, Dense, Dropout, Input


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


def create_small_mlp(
    n_features: int,
    n_categories: int,
    loss: str = "categorical_crossentropy",
    optimizer: str = "adam",
    metrics: List[str] = ["accuracy"],
) -> Sequential:
    """Build and compile a small MLP classifier.

    :param n_features: number of input features (input shape).
    :param n_categories: number of output categories (output shape).
    :param loss:
    :param optimizer:
    :param metrics:
    :return: compiled model.
    """
    model = Sequential()
    model.add(Dense(24, input_dim=n_features, activation="relu"))
    model.add(Dense(48, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(n_categories))

    model.compile(loss=loss, optimizer=optimizer)

    return model


def create_random_mlp(
    n_features: int,
    n_categories: int,
    loss: str = "categorical_crossentropy",
    optimizer: str = "adam",
    metrics: List[str] = ["accuracy"],
) -> Sequential:
    """Build and compile a MLP classifier with heavy dropout.

    :param n_features: number of input features (input shape).
    :param n_categories: number of output categories (output shape).
    :param loss:
    :param optimizer:
    :param metrics:
    :return: compiled model.
    """
    input_layer = Input(shape=(n_features, ))

    hidden_layer_1 = Dense(128, input_shape=(n_features, ), activation="relu")(input_layer)
    dropout_1 = Dropout(0.5)(hidden_layer_1)
    hidden_layer_2 = Dense(256, activation="relu")(dropout_1)
    dropout_2 = Dropout(0.5)(hidden_layer_2)
    hidden_layer_3 = Dense(512, activation="relu")(dropout_2)
    dropout_3 = Dropout(0.5)(hidden_layer_3)
    hidden_layer_4 = Dense(256, activation="relu")(dropout_3)
    dropout_4 = Dropout(0.5)(hidden_layer_4)
    hidden_layer_5 = Dense(128, activation="relu")(dropout_4)
    dropout_5 = Dropout(0.5)(hidden_layer_5)
    output_layer = Dense(n_categories, activation="softmax")(dropout_5)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def create_actor_mlp(
    n_features: int,
    n_categories: int,
) -> Tuple[Input, Sequential]:
    """Build and compile a MLP classifier with 6 layers and dropout.

    :param n_state_features:
    :param n_categories:
    :param batch_size:
    :param loss:
    :param optimizer:
    :param metrics:
    :return:
    """
    model_input = Input(shape=(n_features,))
    hidden_layer_1 = Dense(24, activation='relu')(model_input)
    hidden_layer_2 = Dense(48, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(24, activation='relu')(hidden_layer_2)
    output_layer = Dense(n_categories)(hidden_layer_3)

    model = Model(inputs=model_input, outputs=output_layer)

    return model


def create_critic_mlp(
    n_state_features: int,
    n_action_features: int,
    loss: Any = "mse",
    optimizer: Any = "adam",
) -> Tuple[Input, Input, Sequential]:
    """Build and compile a MLP classifier with 6 layers and dropout.

    :param n_state_features:
    :param n_action_features:
    :param batch_size:
    :param loss:
    :param optimizer:
    :param metrics:
    :return:
    """
    state_input = Input(shape=(n_state_features,))
    state_h1 = Dense(24, activation='relu')(state_input)
    state_h2 = Dense(48, activation='relu')(state_h1)

    action_input = Input(shape=(n_action_features,))
    action_h1 = Dense(48, activation='relu')(action_input)

    state_action = Add()([state_h2, action_h1])
    state_action_h1 = Dense(24, activation='relu')(state_action)
    model_output = Dense(1, activation='linear')(state_action_h1)

    model = Model(inputs=[state_input, action_input], outputs=model_output)
    model.compile(loss=loss, optimizer=optimizer)

    return model
