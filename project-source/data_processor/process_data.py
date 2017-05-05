import pandas as pd
import pickle

from data_processor.data_preprocessor import clean_str, tokenize

class DataProcessor:

    def __init__(self, filePath):
        self.path = filePath
        self.data_relevant = []
        self.data_relavant_tokenized = []
        self.sen_tokens = []
        self.word2vec = []

    def __create_tokens(self):
        data_relavant_tokenized = []
        for index,row in self.data_relevant.iterrows():
            tokens = tokenize(row['text'])
            data_relavant_tokenized.append((tokens, row['stars']))
            self.sen_tokens.append(tokens)
        return data_relavant_tokenized

    def getFormatedData(self, data):
        formattedData = []
        for row in data:
            star = row[1]
            if star <= 2.5:
                formattedData.append((row[0], 'negative'))
            else:
                formattedData.append((row[0], 'positive'))
        return formattedData

    def load_data(self):
        print("Loading data_processor............")
        data = pd.read_csv(self.path, header=0)
        self.data_relevant = data[['text', 'stars']]
        print("Data loaded and relevant columns extracted..........")
        self.data_relavant_tokenized = self.__create_tokens()
        print("Review tokens generated..............")

    def save(self, filepath):
        """
        Save the parameters with pickle
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
