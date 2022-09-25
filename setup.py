import tensorflow_text as tf_text
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ciphers import *

def one_hot_encoding(word, uniques):
    N = word.shape[0]
    enc = np.zeros((N, uniques.shape[0]))
    for i in range(N):
        enc[i] = (word[i] == uniques)
    return enc

def build_model(max_word_length, num_uniques, units, learning_rate):
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences = True), input_shape = (max_word_length, num_uniques)),
        Bidirectional(LSTM(32, return_sequences = True)),
        Dense(num_uniques, activation = 'softmax')
    ])

    model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def decode_preds(preds, uniques):
    func = np.vectorize(chr)
    chars = preds.argmax(axis = 2)
    words = func(uniques[chars]).tolist()
    words = np.array([''.join(word) for word in words], dtype = str)

    return words

def prepare_and_build_model(cipher, key, batch_size = 32, learning_rate = 0.001, validation_split = 0.2, epochs = 200, units = 64):
    words = np.loadtxt('words_alpha.txt', dtype = str)
    idx = np.random.permutation(words.shape[0])
    words = words[idx][:150000]

    M = words.shape[0]
    max_word_length = len(max(words, key = len))

    if cipher == 1:
        encrypt = np.vectorize(railfence)
    elif cipher == 2:
        encrypt = np.vectorize(irreg_columnar)
    elif cipher == 3:
        encrypt = np.vectorize(caesar)
    elif cipher == 4:
        encrypt = np.vectorize(beaufort)
    elif cipher == 5:
        encrypt = np.vectorize(autokey)
    elif cipher == 6:
        encrypt = np.vectorize(hill)

    np.random.seed(100)
    idx = np.random.permutation(M)
    words = words[idx]
    words_enc = encrypt(words, key)

    tokenizer = tf_text.UnicodeCharTokenizer()
    X_tokens = tokenizer.tokenize(words_enc).to_list()
    y_tokens = tokenizer.tokenize(words).to_list()

    X_pad = pad_sequences(X_tokens, maxlen = max_word_length, padding = 'post', truncating = 'post')
    y_pad = pad_sequences(y_tokens, maxlen = max_word_length, padding = 'post', truncating = 'post')

    uniques = np.unique(y_pad)
    num_uniques = uniques.shape[0]

    X = np.zeros((M, max_word_length, num_uniques))
    y = np.zeros((M, max_word_length, num_uniques))
    for i in range(M):
        X[i] = one_hot_encoding(X_pad[i], uniques)
        y[i] = one_hot_encoding(y_pad[i], uniques)

    split = int(M * 0.8)
    train_x, test_x = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    model = build_model(max_word_length, num_uniques, units, learning_rate)
    history = model.fit(train_x, train_y, validation_split = validation_split, epochs = epochs, use_multiprocessing = True, batch_size = batch_size, verbose = 0)

    predictions = model.predict(test_x, verbose = 0)
    preds = decode_preds(predictions, uniques)
    true = decode_preds(test_y, uniques)
    acc = np.sum(preds == true) / true.shape[0]
    return model, history, acc, preds, true

def polycipher_build(cipher, key, batch_size = 32, learning_rate = 0.001, validation_split = 0.2, epochs = 200, units = 64):
    words = np.loadtxt('words_alpha.txt', dtype = str)
    idx = np.random.permutation(words.shape[0])
    words = words[idx][:150000]

    M = words.shape[0]
    max_word_length = len(max(words, key = len))

    if cipher == 1:
        e1 = np.vectorize(autokey)
        e2 = np.vectorize(irreg_columnar)
    elif cipher == 2:
        e1 = np.vectorize(hill)
        e2 = np.vectorize(railfence)
    elif cipher == 3:
        e1 = np.vectorize(hill)
        e2 = np.vectorize(beaufort)
    elif cipher == 4:
        e1 = np.vectorize(autokey)
        e2 = np.vectorize(hill)
    elif cipher == 5:
        e1 = np.vectorize(railfence)
        e2 = np.vectorize(beaufort)
    elif cipher == 6:
        encrypt = np.vectorize(advanced_sub)

    np.random.seed(100)
    idx = np.random.permutation(M)
    words = words[idx]
    if cipher != 6:
        words_enc = e1(words, key[0])
        words_enc = e2(words_enc, key[1])
    else:
        words_enc = encrypt(words, key)

    tokenizer = tf_text.UnicodeCharTokenizer()
    X_tokens = tokenizer.tokenize(words_enc).to_list()
    y_tokens = tokenizer.tokenize(words).to_list()

    X_pad = pad_sequences(X_tokens, maxlen = max_word_length, padding = 'post', truncating = 'post')
    y_pad = pad_sequences(y_tokens, maxlen = max_word_length, padding = 'post', truncating = 'post')

    uniques = np.unique(y_pad) # X_pad
    num_uniques = uniques.shape[0]

    X = np.zeros((M, max_word_length, num_uniques))
    y = np.zeros((M, max_word_length, num_uniques))
    for i in range(M):
        X[i] = one_hot_encoding(X_pad[i], uniques)
        y[i] = one_hot_encoding(y_pad[i], uniques)

    split = int(M * 0.8)
    train_x, test_x = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    model = build_model(max_word_length, num_uniques, units, learning_rate)
    history = model.fit(train_x, train_y, validation_split = validation_split, epochs = epochs, use_multiprocessing = True, batch_size = batch_size, verbose = 0)

    predictions = model.predict(test_x, verbose = 0)
    preds = decode_preds(predictions, uniques)
    true = decode_preds(test_y, uniques)
    acc = np.sum(preds == true) / true.shape[0]
    return model, history, acc, preds, true