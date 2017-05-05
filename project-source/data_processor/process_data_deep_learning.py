import numpy as np
import pandas as pd
import itertools
from collections import Counter
from sklearn.model_selection import train_test_split
from data_processor.data_preprocessor import clean_str, tokenize
import pickle
"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""
class DeepDataProcessor:

    def __init__(self, path):
        self.path = path
        self.data = []
        self.examples = []
        self.labels = []
        self.output = []

    def load_file_and_create_positive_and_negative_example(self):
        all_data = pd.read_csv(self.path, header=0, encoding='utf8')
        self.data = all_data[['text', 'stars']]
        for index, row in self.data.iterrows():
            text = row['text'].lower()
            stars = row['stars']
            if stars <= 2.5:
                self.examples.append(text)
                self.labels.append(0)
            else:
                self.examples.append(text)
                self.labels.append(1)

    def load_data_and_labels(self):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        self.load_file_and_create_positive_and_negative_example()

        # Split by words
        x_text = [s.strip() for s in self.examples]
        x_text = [clean_str(sent) for sent in x_text]
        x_text = [tokenize(s) for s in x_text]

        # Generate labels
        y = np.array(self.labels)

        return [x_text, y]

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word

        #uncomment for rnn
        #vocabulary_inv = [x[0] for x in word_counts.most_common(4999)]

        #comment for rnn
        vocabulary_inv = [x[0] for x in word_counts.most_common()]

        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]


    def build_input_data(self, sentences, labels, vocabulary):
        """
        Maps sentencs and labels to vectors based on a vocabulary.
        """
        #uncomment for rnn
        #x = np.array([[vocabulary.get(word, 5000) for word in sentence] for sentence in sentences])

        #comment for rnn
        x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
        y = np.array(labels)
        return [x, y]


    def load_data(self):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        sentences, labels = self.load_data_and_labels()

        #comment this for RNN, and pass senteces as parameter in build_vocab and build_input_data
        #sentences_padded = self.pad_sentences(sentences)
        vocabulary, vocabulary_inv = self.build_vocab(sentences)
        x, y = self.build_input_data(sentences, labels, vocabulary)
        data = [(x[i], y[i]) for i in range(len(x))]
        data_train, data_test = train_test_split(data, test_size=0.2)
        x_train = np.array([d[0] for d in data_train])
        x_test = np.array([d[0] for d in data_test])
        y_train = np.array([d[1] for d in data_train])
        y_test = np.array([d[1] for d in data_test])
        self.output = [x_train,x_test, y_train, y_test, vocabulary, vocabulary_inv]

    def save(self, filepath):
        """
        Save the parameters with pickle
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

