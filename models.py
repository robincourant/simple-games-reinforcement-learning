from keras.models import Sequential
from keras.layers import Dense, Dropout


def create_MLP_model(
    n_features: int,
    n_categories: int,
) -> Sequential:
    """Build and compile a MLP with 6 layers and dropout.

    :param n_features: number of input features (input shape).
    :param n_categories: number of output categories (output shape).
    :return: a compiled model.
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

    model.add(Dense(n_categories, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])

    return model
