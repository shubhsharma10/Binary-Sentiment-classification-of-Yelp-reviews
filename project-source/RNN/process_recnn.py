# LSTM for sequence classification in the IMDB dataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from data_processor.word2vec import train_word2vec


def run_rnn(process_data, model_type="non-static"):
    # fix random seed for reproducibility
    np.random.seed(7)
    max_review_length = 400
    embedding_vector_length = 300
    lstm_layers = 100
    dropout_prob = (0.5, 0.8)
    batch_size=64
    no_of_epochs = 4

    # load the dataset but only keep the top n words, zero the rest
    X_train, X_test,y_train, y_test, vocab,vocab_inverse=process_data.output
    # truncate and pad input sequences

    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    print("Model type is", model_type)
    if model_type == "non-static":
        embedding_weights = train_word2vec(np.vstack((X_train, X_test)), vocab_inverse, num_features=embedding_vector_length,
                                       min_word_count=1, context = 10)
    else:
        embedding_weights = None

    model = Sequential()
    model.add(Embedding(len(vocab_inverse), embedding_vector_length, input_length=max_review_length, weights=embedding_weights))
    model.add(Dropout(dropout_prob[0]))
    model.add(LSTM(lstm_layers))
    model.add(Dropout(dropout_prob[1]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=no_of_epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    # Final evaluation of the model

    scores = model.evaluate(X_test, y_test, verbose=0)

    print("Accuracy: %.2f%%" % (scores[1]*100))


