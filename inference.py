from preprocessing import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model


class LSTMInference:
    def __init__(self, model_path, data, embedding_dim, lstm_units):
        self.x_vocab_size, self.y_vocab_size = data.get_vocab_size()
        self.max_decoder_seq_length = data.y_len
        self.input_seq_length = data.x_len
        self.x_tokenizer = data.x_tk
        self.y_tokenizer = data.y_tk
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

        # Load the trained model
        self.model = load_model(model_path)

        # Extract encoder and decoder from the trained model
        self.encoder_model, self.decoder_model = self.build_inference_model()

    def build_inference_model(self):
        # Encoder model
        encoder_inputs = self.model.input[0]  # encoder input from the trained model
        enc_emb = self.model.layers[2](encoder_inputs)
        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[4](enc_emb)  # encoder LSTM output states
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

        # Decoder model
        decoder_inputs = self.model.input[1]  # decoder input from the trained model
        decoder_states_input_h = Input(shape=(self.lstm_units,), name='input_h')
        decoder_states_input_c = Input(shape=(self.lstm_units,), name='input_c')
        decoder_states_inputs = [decoder_states_input_h, decoder_states_input_c]
        dec_emb_layer = self.model.layers[3]
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_lstm = self.model.layers[5]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]

        # Output layer
        dense_layer = self.model.layers[6]
        dense_output = dense_layer(decoder_outputs)

        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [dense_output] + decoder_states
        )

        return encoder_model, decoder_model

    def decode_sequence(self, sentence):
        input_seq = self.x_tokenizer.texts_to_sequences([sentence])
        input_seq = pad_sequences(input_seq, maxlen=self.input_seq_length, padding='post')
        # Encode the input sequence to get the initial states
        encoder_outputs, state_h, state_c = self.encoder_model.predict(input_seq)
        states_value = [state_h, state_c]

        # Generate empty target sequence of length 1
        target_seq = np.zeros((1, 1))

        # Populate the first character of target sequence with the start token
        target_seq[0, 0] = self.y_tokenizer.word_index['<s>']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.y_tokenizer.index_word[sampled_token_index]
            decoded_sentence += ' ' + sampled_char

            # Exit condition: either hit max length or find stop token
            if (sampled_char == '<e>' or len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return decoded_sentence


class AttentionLSTMInference:
    def __init__(self, model_path, data, embedding_dim, lstm_units):
        self.x_vocab_size, self.y_vocab_size = data.get_vocab_size()
        self.max_decoder_seq_length = data.y_len
        self.input_seq_length = data.x_len
        self.x_tokenizer = data.x_tk
        self.y_tokenizer = data.y_tk
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

        # Load the trained model
        self.model = load_model(model_path)

        # Extract encoder and decoder from the trained model
        self.encoder_model, self.decoder_model = self.build_inference_model()

    def build_inference_model(self):
        # Encoder model
        encoder_inputs = self.model.input[0]  # encoder input from the trained model
        enc_emb = self.model.layers[2](encoder_inputs)
        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[4](enc_emb)  # encoder LSTM output states
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

        # Decoder model
        decoder_inputs = self.model.input[1]  # decoder input from the trained model
        decoder_states_input_h = Input(shape=(self.lstm_units,), name='input_h')
        decoder_states_input_c = Input(shape=(self.lstm_units,), name='input_c')
        decoder_states_inputs = [decoder_states_input_h, decoder_states_input_c]
        dec_emb_layer = self.model.layers[3]
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_lstm = self.model.layers[5]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]

        # Attention mechanism
        attention = self.model.layers[6]
        attention_result = attention([decoder_outputs, encoder_outputs])
        decoder_concat_input = tf.concat([decoder_outputs, attention_result], axis=-1)

        # Output layer
        output_layer = self.model.layers[8]
        decoder_outputs = output_layer(decoder_concat_input)

        decoder_model = Model(
            [decoder_inputs] + [encoder_outputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        return encoder_model, decoder_model

    def decode_sequence(self, sentence):
        input_seq = self.x_tokenizer.texts_to_sequences([sentence])
        input_seq = pad_sequences(input_seq, maxlen=self.input_seq_length, padding='post')
        # Encode the input sequence to get the initial states
        encoder_outputs, state_h, state_c = self.encoder_model.predict(input_seq)
        states_value = [state_h, state_c]

        # Generate empty target sequence of length 1
        target_seq = np.zeros((1, 1))

        # Populate the first character of target sequence with the start token
        target_seq[0, 0] = self.y_tokenizer.word_index['<s>']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + [encoder_outputs] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.y_tokenizer.index_word[sampled_token_index]
            decoded_sentence += ' ' + sampled_char

            # Exit condition: either hit max length or find stop token
            if (sampled_char == '<e>' or len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return decoded_sentence


def main(data_path, model_path):
    set_seed(1)
    set_gpu()

    preprocessor = Preprocessor(data_path)
    print(preprocessor)
    print("----------Model Selection----------")
    while 1:
        model_type = str(input('Type the type of model(lstm or attention_lstm): '))
        if model_type == 'attention_lstm':
            attention_lstm_model_path = model_path + "/AttentionLSTM.h5"
            inference = AttentionLSTMInference(attention_lstm_model_path, preprocessor, embedding_dim=16, lstm_units=32)
            break
        elif model_type == 'lstm':
            lstm_model_path = model_path + "/LSTM.h5"
            inference = LSTMInference(lstm_model_path, preprocessor, embedding_dim=16, lstm_units=32)
            break

    print("----------Translation----------")
    while 1:
        sentence = str(input('Type an English sentence: '))
        if sentence == 'exit':
            exit(0)
        zh_sentence = inference.decode_sequence(sentence)
        print(f"Translate to Chinese: {zh_sentence}\n")


if __name__ == '__main__':
    main(data_path='./data', model_path='./models')