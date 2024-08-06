import os
os.environ['TF_ENABLED_ONEDNN_OPTS'] = '0'

import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import jieba
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Preprocessor:

    def __init__(self, path):
        self.path = path

        self.input = []
        self.target = []
        self.tokenizer_input = Tokenizer()
        self.tokenizer_target = Tokenizer()

        self.input_sequences = None
        self.target_sequences = None
        self.pred = None
        self.pred_sequences = None

        self.input_vocab_size = None
        self.target_vocab_size = None

        self.max_input_length = None
        self.max_target_length = None

    def __str__(self):
        input_sequences = f"Input sequences: {self.input_sequences.shape}\n"
        target_sequences = f"Target sequences: {self.target_sequences.shape}\n"
        return input_sequences + target_sequences

    def load_data(self):
        # with open(f"{self.path}/cmn.txt", 'r', encoding="utf-8") as f:
        #     for line in f.readlines():
        #         input_line, target_line = line.strip().split('\n')[:2]
        #         input_line = self.clean_text(input_line)
        #         target_line = self.clean_text(target_line)
        #         self.input.append(input_line)
        #         self.target.append(target_line)
        #     target_texts_segmented = [' '.join(jieba.cut(text)) for text in self.target]
        #     self.target = target_texts_segmented

        with open(f"{self.path}/news.tsv", newline='', encoding='utf-8') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            for row in reader:
                input_line, target_line = row[0], row[1]
                input_line = self.clean_text(input_line)
                target_line = self.clean_text(target_line)
                self.input.append(input_line)
                self.target.append(target_line)
            target_texts_segmented = [' '.join(jieba.cut(text)) for text in self.target]
            self.target = target_texts_segmented

    def clean_text(self, text):
        text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9\s]', '', text)
        return text

    def tokenize(self, pred=None):
        if pred is None:
            self.tokenizer_input.fit_on_texts(self.input)
            input_sequences = self.tokenizer_input.texts_to_sequences(self.input)

            self.tokenizer_target.fit_on_texts(self.target)
            target_sequences = self.tokenizer_target.texts_to_sequences(self.target)

            self.max_input_length = max([len(seq) for seq in input_sequences])
            self.max_target_length = max([len(seq) for seq in target_sequences])

            self.input_sequences = pad_sequences(input_sequences, maxlen=self.max_input_length, padding='post')
            self.target_sequences = pad_sequences(target_sequences, maxlen=self.max_target_length, padding='post')

            self.input_vocab_size = len(self.tokenizer_input.word_index) + 1
            self.target_vocab_size = len(self.tokenizer_target.word_index) + 1
        else:
            self.pred = pred
            self.tokenizer_input.fit_on_texts(self.pred)
            pred_sequences = self.tokenizer_input.texts_to_sequences(self.pred)
            self.pred_sequences = pad_sequences(pred_sequences, maxlen=self.max_input_length, padding='post')

    def w2v_model(self, threshold=0.5):
        all_sentences = self.input + self.target
        all_sentences = [sentence.split() for sentence in all_sentences]
        w2v_model = Word2Vec(sentences=all_sentences, vector_size=100, alpha=0.001, min_alpha=0.000001, epochs=100,
                             batch_words=5000, seed=1, window=10, workers=20)
        en_vec = np.array([w2v_model.wv[word] for word in self.tokenizer_input.word_index.keys() if word in w2v_model.wv])
        ch_vec = np.array([w2v_model.wv[word] for word in self.tokenizer_target.word_index.keys() if word in w2v_model.wv])

        similarities = cosine_similarity(en_vec, ch_vec)

        pairs = []
        for i in range(len(similarities)):
            for j in range(len(similarities[i])):
                if similarities[i][j] >= threshold:
                    pairs.append((list(self.tokenizer_input.word_index.keys())[i], list(self.tokenizer_target.word_index.keys())[j], similarities[i][j]))

        pairs_df = pd.DataFrame(pairs, columns=['input', 'target', 'similarity'])
        return w2v_model, pairs_df

    def plot_similarities(self, pairs_df, sample_size=50):
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['font.sans-serif'] = ['Arial']
        sampled_pairs = pairs_df.sample(n=sample_size, random_state=0)
        plt.figure(figsize=(20, 20))
        plt.scatter(sampled_pairs['input'], sampled_pairs['target'], c=sampled_pairs['similarity'], cmap='viridis')
        plt.colorbar(label='Similarity')
        plt.xlabel('English')
        plt.xticks(rotation=90)
        plt.ylabel('Chinese')
        plt.title('Similarity Plot')
        plt.savefig('imgs/Similarity.png')
        plt.show()

    def tsne_plot(self, word_vectors, labels, sample_size=100):
        sampled_indices = np.random.choice(len(word_vectors), size=sample_size, replace=False)
        sampled_vectors = word_vectors[sampled_indices]
        sampled_labels = [labels[i] for i in sampled_indices]

        tsne = TSNE(n_components=2, random_state=0)
        reduced_vectors = tsne.fit_transform(sampled_vectors)
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.figure(figsize=(20, 20))
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], edgecolors='k', c='r')
        for label, x, y in zip(sampled_labels, reduced_vectors[:, 0], reduced_vectors[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.savefig('imgs/tsne_plot.png')
        plt.show()


if __name__ == "__main__":
    preprocessor = Preprocessor("data")
    preprocessor.load_data()
    preprocessor.tokenize()
    print(preprocessor)
    w2v_model, pairs = preprocessor.w2v_model(0.95)
    print("-----w2v model created-----")
    preprocessor.plot_similarities(pairs)
    preprocessor.tsne_plot(w2v_model.wv[w2v_model.wv.key_to_index], list(w2v_model.wv.key_to_index.keys()), sample_size=100)
