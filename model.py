import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
from tensorflow.keras.optimizers import Adam


class LSTMModel:
    def __init__(self, data):
        self.x_vocab_size, self.y_vocab_size = data.get_vocab_size()
        self.output_shape = data.y_len

    def plot_history(self):
        plt.figure(figsize=(10, 10))
        # Plot training & validation accuracy values
        plt.plot(self.history.history['accuracy'], label="Training Accuracy")
        plt.plot(self.history.history['val_accuracy'], label="Validation Accuracy")
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('imgs/lstm_accuracy_plot.png')

        # Plot training & validation loss values
        plt.figure(figsize=(10, 10))
        plt.plot(self.history.history['loss'], label="Training Loss")
        plt.plot(self.history.history['val_loss'], label="Validation Loss")
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('imgs/lstm_loss_plot.png')

    def train_model(self, data, embedding_dim, lstm_units, epochs, batch_size):
        encoder_inputs = Input(shape=(None,))
        enc_emb = Embedding(self.x_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
        encoder_lstm = LSTM(lstm_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        dec_emb = Embedding(self.y_vocab_size, embedding_dim, mask_zero=True)
        dec_emb = dec_emb(decoder_inputs)
        decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
        decoder_dense = Dense(self.y_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        train_data, test_data = data.create_dataset(batch_size=batch_size)
        self.history = model.fit(train_data, epochs=epochs, validation_data=test_data)

        # Save model and batch history
        model.save('models/LSTM.h5')
        pd.DataFrame(self.history.history).to_csv('models/LSTM_history.csv', index=False)
        self.plot_history()


class AttentionLSTMModel:
    def __init__(self, data):
        self.x_vocab_size, self.y_vocab_size = data.get_vocab_size()
        self.output_shape = data.y_len

    def plot_history(self):
        plt.figure(figsize=(10, 10))
        # Plot training & validation accuracy values
        plt.plot(self.history.history['accuracy'], label="Training Accuracy")
        plt.plot(self.history.history['val_accuracy'], label="Validation Accuracy")
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('imgs/attention_lstm_accuracy_plot.png')

        # Plot training & validation loss values
        plt.figure(figsize=(10, 10))
        plt.plot(self.history.history['loss'], label="Training Loss")
        plt.plot(self.history.history['val_loss'], label="Validation Loss")
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('imgs/attention_lstm_loss_plot.png')

    def train_model(self, data, embedding_dim, lstm_units, epochs, batch_size):
        encoder_inputs = Input(shape=(None,))
        enc_emb = Embedding(self.x_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
        encoder_lstm = LSTM(lstm_units, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        dec_emb = Embedding(self.y_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

        # Add attention mechanism
        attention_layer = Attention()
        attention_result = attention_layer([decoder_outputs, encoder_outputs])

        # Concatenate the context vector with the decoder outputs
        decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_result])
        decoder_dense = Dense(self.y_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_concat_input)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        optimizer = Adam(learning_rate=0.00001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        train_data, test_data = data.create_dataset(batch_size=batch_size)
        self.history = model.fit(train_data, epochs=epochs, validation_data=test_data)

        # Save model and batch history
        model.save('models/AttentionLSTM.h5')
        pd.DataFrame(self.history.history).to_csv('models/AttentionLSTM_history.csv', index=False)
        self.plot_history()


if __name__ == '__main__':
    set_seed(1)
    set_gpu()

    embedding_dim = 16
    lstm_units = 32
    batch_size = 6
    epochs = 5

    preprocessor = Preprocessor(path='./data')
    print(preprocessor)
    attention_lstm_model = AttentionLSTMModel(preprocessor)
    attention_lstm_model.train_model(preprocessor, embedding_dim, lstm_units, epochs, batch_size)