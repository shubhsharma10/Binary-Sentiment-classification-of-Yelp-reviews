from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from data_processor.data_preprocessor import get_x_vectors, get_y_vectors


def run_svm(data_processor):
    vectorizer = TfidfVectorizer(max_features=5000, use_idf=True)
    print("Vocabulary features loaded............")
    data_relavant_train, data_relavant_test = train_test_split(data_processor.data_relevant, test_size=0.2)
    print("Data split into traning(80%) and test(20%)")

    train_X = vectorizer.fit_transform(get_x_vectors(data_relavant_train))
    test_X = vectorizer.transform(get_x_vectors(data_relavant_test))
    print(train_X.shape)
    train_Y = get_y_vectors(data_relavant_train)
    test_Y = get_y_vectors(data_relavant_train)
    print("Train and test data vector created")
    print("Training Classifier")
    if train_X.shape[0] > 10000:
        print("Using Linear SVM")
        clf = LinearSVC()
    else:
        print("Using Non-linear SVM")
        clf = SVC()

    clf.fit(train_X, train_Y)
    print("Classifier Trained")
    print("Calculating Accuracy")
    prediction = clf.predict(test_X)
    correct = 0
    for index in range(len(prediction)):
        if prediction[index] == test_Y[index]:
            correct += 1
    print('Accuracy', ((correct / len(data_relavant_test)) * 100))
