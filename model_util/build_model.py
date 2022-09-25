import tensorflow_text as tf_text
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Iterable, Any
from model_util.ciphers import *

def one_hot_encoding(word, uniques):
    # OHE plaintext words
    
    N = word.shape[0]
    enc = np.zeros((N, uniques.shape[0]))
    for i in range(N):
        enc[i] = (word[i] == uniques)
    return enc

def build_model(max_word_length, num_uniques, units, learning_rate):
    # Construct RNN with given architecture
    
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences = True), input_shape = (max_word_length, num_uniques)),
        Bidirectional(LSTM(32, return_sequences = True)),
        Dense(num_uniques, activation = 'softmax')
    ])

    model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def decode_preds(preds, uniques):
    # Decode the softmax predictions of RNN
    
    func = np.vectorize(chr)
    chars = preds.argmax(axis = 2)
    words = func(uniques[chars]).tolist()
    words = np.array([''.join(word) for word in words], dtype = str)

    return words

# Store monociphers
monociphers = {
    1: np.vectorize(railfence),
    2: np.vectorize(irreg_columnar),
    3: np.vectorize(caesar),
    4: np.vectorize(beaufort),
    5: np.vectorize(autokey),
    6: np.vectorize(hill)
}

# Store polyciphers + paper cipher
polyciphers = {
    1: (np.vectorize(autokey), np.vectorize(irreg_columnar)),
    2: (np.vectorize(hill), np.vectorize(railfence)),
    3: (np.vectorize(hill), np.vectorize(beaufort)),
    4: (np.vectorize(autokey), np.vectorize(hill)),
    5: (np.vectorize(railfence), np.vectorize(beaufort)),
    6: np.vectorize(advanced_sub)
}

def build_and_train_model(choice: tuple[int, int], key: Iterable[Any], frac_words: float = 0.5, batch_size: int = 32, learning_rate: float = 0.001, validation_split: float = 0.2, test_size: float = 0.2, epochs: int = 5, units: int = 64, seed: int = 100) -> tuple:
    """
    Encrypts, tokenizes, and pads plaintext words from words_alpha.txt.
    Constructs training and test data, then trains RNN on training data.
    Returns the trained model, model history, word accuracy, predicted words, and true words.
    """
    
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    
    words = np.loadtxt('words_alpha.txt', dtype = str)
    total_words = words.shape[0]
    idx = np.random.permutation(total_words)
    words = words[idx][:int(total_words * frac_words)]

    M = words.shape[0]
    max_word_length = len(max(words, key = len))

    if choice[0] == 0:
        encrypt = monociphers[choice[1]]
        words_enc = encrypt(words, key)
    elif choice[0] == 1:
        if choice[1] != 6:
            e1, e2 = polyciphers[choice[1]]
            words_enc = e1(words, key[0])
            words_enc = e2(words_enc, key[1])
        else:
            words_enc = encrypt(words, key)
    else:
        raise Exception('Invalid cipher type (choose 0/1)')

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

    split = int(M * (1 - test_size))
    train_x, test_x = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    model = build_model(max_word_length, num_uniques, units, learning_rate)
    history = model.fit(train_x, train_y, validation_split = validation_split, epochs = epochs, use_multiprocessing = True, batch_size = batch_size, verbose = 0)

    predictions = model.predict(test_x, verbose = 0)
    preds = decode_preds(predictions, uniques)
    true = decode_preds(test_y, uniques)
    acc = np.sum(preds == true) / true.shape[0]
    return model, history, acc, preds, true