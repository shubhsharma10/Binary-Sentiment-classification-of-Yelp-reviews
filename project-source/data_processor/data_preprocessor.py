import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def tokenize(sentence):
    tokens = re.findall('[a-z]+', sentence.lower())
    tokens = [clean_str(t) for t in tokens]
    new_tokens = []
    for t in tokens:
        if len(t) > 2:
            new_tokens.append(t)
    return new_tokens

def get_x_vectors(data_relavant):
    content = []
    for index,data in data_relavant.iterrows():
        content.append(data['text'])
    return content

def get_y_vectors(data_relavant):
    content = []
    for index,data in data_relavant.iterrows():
        star = data['stars']
        if star >= 2.5:
            content.append(0)
        else:
            content.append(1)
    return content