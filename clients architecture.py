from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, GRU, SimpleRNN,
    Conv1D, MaxPooling1D, Flatten
)

def compile_model(model):

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_model(model_name, input_shape, num_classes):

    model = Sequential()

    if model_name == "MLP":
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))

    elif model_name == "LSTM":
        model.add(LSTM(64, input_shape=input_shape))
        model.add(Dropout(0.3))

    elif model_name == "GRU":
        model.add(GRU(64, input_shape=input_shape))
        model.add(Dropout(0.3))

    elif model_name == "RNN":
        model.add(SimpleRNN(64, input_shape=input_shape))

    elif model_name == "CNN":
        model.add(Conv1D(64,3,activation='relu',input_shape=input_shape))
        model.add(MaxPooling1D(2))
        model.add(Flatten())

    elif model_name == "CNN_LSTM":
        model.add(Conv1D(64,3,activation='relu',input_shape=input_shape))
        model.add(MaxPooling1D(2))
        model.add(LSTM(64))

    elif model_name == "CNN_GRU":
        model.add(Conv1D(64,3,activation='relu',input_shape=input_shape))
        model.add(MaxPooling1D(2))
        model.add(GRU(64))

    elif model_name == "CNN_RNN":
        model.add(Conv1D(64,3,activation='relu',input_shape=input_shape))
        model.add(MaxPooling1D(2))
        model.add(SimpleRNN(64))

    elif model_name == "LSTM_GRU":
        model.add(LSTM(64,return_sequences=True,input_shape=input_shape))
        model.add(GRU(64))

    elif model_name == "LSTM_RNN":
        model.add(LSTM(64,return_sequences=True,input_shape=input_shape))
        model.add(SimpleRNN(64))

    else:
        raise ValueError("Unknown model")

    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return compile_model(model)
