import nltk as nl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from data_processor.data_preprocessor import get_x_vectors, get_y_vectors

vocab_global = []
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in vocab_global:
        features['contains(%s)' % word] = (word in document_words)
    return features

def run_naive_bayes(data_processor):
    vectorizer = TfidfVectorizer(max_features=5000, use_idf=True)
    vectorizer.fit_transform(get_x_vectors(data_processor.data_relevant))
    vocab_global = vectorizer.vocabulary_.keys()
    print("Vocabulary features loaded............")
    data_relavant_train, data_relavant_test = train_test_split(data_processor.data_relavant_tokenized, test_size=0.2)
    print("Data split into traning(80%) and test(20%)")
    training_set = nl.classify.apply_features(extract_features, data_processor.getFormatedData(data_relavant_train))
    print("Training dataset created in correct format")
    classifier = nl.NaiveBayesClassifier.train(training_set)
    print("Trained Naive Bayes classifier")
    print("Calulating Accuracy")
    correct = 0
    for row in data_processor.getFormatedData(data_relavant_test):
        prediction = classifier.classify(extract_features(row[0]))
        if prediction == row[1]:
            correct += 1
    print('Accuracy', ((correct / len(data_relavant_test)) * 100))
