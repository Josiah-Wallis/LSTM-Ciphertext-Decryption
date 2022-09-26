# Supervised LSTM Ciphertext Decryption
In the field of **cryptography**, **encryption algorithms** are used to transform plaintext strings into altered strings of variable size. This process **increases** the **security** of **information** being transmitted across communication channels. Here, I present a **proof of concept** that **supervised LSTM** recurrent neural networks (RNN) are **more efficient** at deciphering plaintext ciphers **than typical** cryptographic and cryptanalysis **techniques**. I also present an **LSTM's performance** on a [complex encryption algorithm][paper cipher] conjectured in 2014 with **fruitful results**.

# Implementation
### Overview
For every cipher and polycipher I used to encrypt my [data][dataset], a list of English plaintext words, I trained a 2-stack bidirectional LSTM to decrypt words enciphered by the respective input cipher. The pipeline for this process follows:
1) Encipher the dataset using one cipher
2) Tokenize and pad data for RNN
3) Train RNN
4) Test RNN on words enciphered by cipher in step 1

Performance was judged using validation accuracy (character-to-character) and the RNN's accuracy on the test set (word-to-word). 

### Package Versions
The setup was built using:
* Python 3.10.6
* Numpy 1.23.2
* Tensorflow 2.9.1
* Keras 2.9.0
* Tensorflow Text 2.10.0

### Usage
To quickly get started training, below is a quick-start usage case:
```
from model_util.build_model import build_and_train_model

''' 
# choice[0] == 0
monociphers = {
    1: np.vectorize(railfence),
    2: np.vectorize(irreg_columnar),
    3: np.vectorize(caesar),
    4: np.vectorize(beaufort),
    5: np.vectorize(autokey),
    6: np.vectorize(hill)
}

# choice[0] == 1
polyciphers = {
    1: (np.vectorize(autokey), np.vectorize(irreg_columnar)),
    2: (np.vectorize(hill), np.vectorize(railfence)),
    3: (np.vectorize(hill), np.vectorize(beaufort)),
    4: (np.vectorize(autokey), np.vectorize(hill)),
    5: (np.vectorize(railfence), np.vectorize(beaufort)),
    6: np.vectorize(advanced_sub)
}
'''

def main():
    choice = (0, 1)
    keys = 2
    frac_words = 0.4
    model, history, acc, prediction, true_label = build_and_train_model(choice, keys, frac_words = frac_words, epochs = 5, units = 64, batch_size = 36)
```
For full implementation and documentation details of the arguments and methods used, please refer to the commented code files. An in-depth demonstration is provided in LSTM_Ciphertext_Decryption.ipynb.

# Results
Provided below are the validation accuracies of RNNs trained on data enciphered by polyciphers. In each case, the RNNs achieve over 96% validation accuracy (character-to-character). 
![output1.png](https://www.dropbox.com/s/nqtf8f8m0vn1ms6/output1.png?dl=0&raw=1)
Additionally, the LSTM architecture performs very well on the plaintext words encrypted using the [2014 cipher][paper cipher], obtaining a validation accuracy of almost 100%.
![output2.png](https://www.dropbox.com/s/nd13kek940uulx5/output2.png?dl=0&raw=1)

# Conclusion
Because the validation accuracy isn't 100%, sometimes characters are predicted incorrectly thus decreasing the word-to-word accuracy. In cases of slightly mismatched lettering, the correct word can be easily inferred from the predicted word. All in all, the results demonstrate this method of cryptanalysis provides a new frontier by which cryptologists may approach decryption.

# Contact
Please send any questions, comments, or inquiries to jwall014@ucr.edu. A paper rough draft can be found [here][link to my paper].

[link to my paper]: <https://drive.google.com/file/d/1H54A94rNeenz52AYRwqPaJc_klwU9kmQ/view?usp=sharing>

[dataset]: <https://github.com/dwyl/english-words/blob/master/words_alpha.txt>

[paper cipher]: <https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.429.1120>