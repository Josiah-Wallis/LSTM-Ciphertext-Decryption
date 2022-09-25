Four files are included:
1) analysis.ipynb: Contains the code and analysis I performed for the paper
2) LSTM_Ciphertext_Decryption.ipynb: I simple demonstration of my code with metadata output
3) ciphers.py: Contains all the encryption algorithms I implemented
4) setup.py: contains the model details, preprocessing, and the methods I used to generate my dataset

** Details for (1) 
No running is required, but you may see my results. You may also rerun the notebook to see my results, but they may vary a little bit as I did not seed my permutation of the datasets when they were created. The results should still be very similar. You can also find the hyperparameters I used for each model as well as the keys for each encryption algorithm.

** Details for (2)
A broken-down demonstration of my code is given. It trains an LSTM on railfence-enciphered words. The validation accuracy is given as character-by-character accuracy, while the accuracy at the end of the notebook is word-by-word accuracy

** Details for (3) 
Every cipher I used as well as how they work are all presented, including the cipher I used from the research paper. 

** Details for (4)
The details of how I made my dataset, how I implemented the LSTM, and how I encode/decode inputs/outputs are given here.

words_alpha is the dataset I used from the github link referenced in the paper