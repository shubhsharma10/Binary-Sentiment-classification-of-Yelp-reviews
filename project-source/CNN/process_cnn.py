
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from data_processor.word2vec import train_word2vec, load_google_trained_w2v
import numpy as np

'''
Original from https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
'''
def run_cnn(deep_data_processor, model_type):
    x_train, x_test, y_train, y_test, vocabulary, vocabulary_inv = deep_data_processor.output

    #w2v_type = "trained" # trained | pre-trained
    # Model Hyperparameters
    embedding_dim = 300 # 50 | 300
    filter_sizes = (3,4,5)
    num_filters = 10
    dropout_prob = (0.5, 0.8)
    hidden_dims = 100

    # Training parameters
    batch_size = 64
    num_epochs = 4

    # Prepossessing parameters
    sequence_length = 400

    # Word2Vec parameters (see train_word2vec)
    min_word_count = 1
    context = 10

    x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
    x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

     # Prepare embedding layer weights and convert inputs for static model
    print("Model type is", model_type)
    if model_type == "non-static":
        #if w2v_type == "trained":
        embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                           min_word_count=min_word_count, context=context)
        # elif w2v_type == "pre-trained":
        #     embedding_weights = load_google_trained_w2v(vocabulary_inv, embedding_dim)

    elif model_type == "rand":
        embedding_weights = None
    else:
        raise ValueError("Unknown model type")

    # Build model
    input_shape = (sequence_length,)
    model_input = Input(shape=input_shape)

    # Static model do not have embedding layer
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)
    z = Dropout(dropout_prob[0])(z)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Initialize weights with word2vec
    if model_type == "non-static":
        embedding_layer = model.get_layer("embedding")
        embedding_layer.set_weights(embedding_weights)

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
              validation_data=(x_test, y_test))

    scores = model.evaluate(x_test, y_test, verbose=0)

    print("Accuracy: %.2f%%" % (scores[1] * 100))

