import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Flatten
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

INPUT_DIM = X_scaled.shape[1]
NUM_CLASSES = len(np.unique(y_encoded))

# ------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------
def create_model(model_type):

    if model_type == "MLP":
        model = Sequential([
            Dense(128, activation='relu', input_shape=(INPUT_DIM,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    elif model_type == "CNN":
        model = Sequential([
            tf.keras.layers.Reshape((INPUT_DIM,1), input_shape=(INPUT_DIM,)),
            Conv1D(64,1,activation='relu'),
            MaxPooling1D(1),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    elif model_type == "LSTM":
        model = Sequential([
            tf.keras.layers.Reshape((1,INPUT_DIM), input_shape=(INPUT_DIM,)),
            LSTM(64),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    elif model_type == "GRU":
        model = Sequential([
            tf.keras.layers.Reshape((1,INPUT_DIM), input_shape=(INPUT_DIM,)),
            GRU(64),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    elif model_type == "RNN":
        model = Sequential([
            tf.keras.layers.Reshape((1,INPUT_DIM), input_shape=(INPUT_DIM,)),
            SimpleRNN(64),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    elif model_type == "CNN_LSTM":
        model = Sequential([
            tf.keras.layers.Reshape((1,INPUT_DIM), input_shape=(INPUT_DIM,)),
            Conv1D(32,1,activation='relu'),
            LSTM(64),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    elif model_type == "CNN_GRU":
        model = Sequential([
            tf.keras.layers.Reshape((1,INPUT_DIM), input_shape=(INPUT_DIM,)),
            Conv1D(32,1,activation='relu'),
            GRU(64),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    elif model_type == "LSTM_GRU":
        model = Sequential([
            tf.keras.layers.Reshape((1,INPUT_DIM), input_shape=(INPUT_DIM,)),
            LSTM(64, return_sequences=True),
            GRU(32),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
