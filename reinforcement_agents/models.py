import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input


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

    model.save(file_path)
    print(f"Model saved at {file_path}.")


def create_small_mlp(
    n_features: int,
    n_categories: int,
    loss: str = "categorical_crossentropy",
    optimizer: str = "adam",
) -> Sequential:
    """Build and compile a small MLP classifier.

    :param n_features: number of input features (input shape).
    :param n_categories: number of output categories (output shape).
    :param loss: loss used to train the model.
    :param optimizer: optimizer used to train the model.
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
) -> Sequential:
    """Build and compile a MLP classifier with heavy dropout.

    :param n_features: number of input features (input shape).
    :param n_categories: number of output categories (output shape).
    :param loss: loss used to train the model.
    :param optimizer: optimizer used to train the model.
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
    model.compile(loss=loss, optimizer=optimizer)

    return model


def create_actor_critic_mlp(
    n_state_features: int,
    n_action_features: int,
) -> Sequential:
    """Build a non-compiled actor-critic network: 1 input and 2 outputs.

    :param n_state_features: number of state features (input shape).
    :param n_state_features: number of action features (action output shape).
    :return: raw model.
    """
    state_input = Input(shape=(n_state_features,))
    common_layers = Dense(128, activation="relu")(state_input)
    actor_output = Dense(n_action_features, activation="softmax")(common_layers)
    critic_output = Dense(1)(common_layers)

    model = Model(inputs=state_input, outputs=[actor_output, critic_output])

    return model
