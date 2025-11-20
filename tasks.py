import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.datasets import macrodata

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.

# [A] List Comprehensions and String Manipulation: Tokenization

# Task 1: Given a paragraph of text, implement a simple "tokenizer" that splits the paragraph 
#   into individual words (tokens) and removes any punctuation. Implement this using a list 
#   comprehension.

# Your code here:
# -----------------------------------------------
text = "The quick brown fox jumps over the lazy dog!"

# Write a list comprehension to tokenize the text and remove punctuation
tokens = [word.strip("!.,") for word in text.split(" ")]

# Expected output: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
print(tokens)
# -----------------------------------------------


# Task 2: Create a function that takes a string and breaks it up into tokens and removes any 
#   punctuation, and then converts each token to lowercase. 
#   (NOTE: Function modified to preserve order/duplicates for later Tasks 3, 4, and 8)

# Your code here:
# -----------------------------------------------
def tokenize(string: str) -> list:
    # Returns full list of tokens to allow frequency counting and sequence modeling
    return [word.strip('!.,?').lower() for word in string.split() if word.strip('!.,?')]

# -----------------------------------------------


# [B] Dictionary Comprehensions: Frequency Count of Tokens

# Task 3: Using the tokens list from the previous exercise, create a dictionary comprehension 
#   that counts the frequency of each word.

# Your code here:
# -----------------------------------------------
# We iterate over set(tokenize(text)) to avoid recounting the same word multiple times
word_frequencies = {word: tokenize(text).count(word) for word in set(tokenize(text))}

# Expected output example: {'the': 2, 'quick': 1, ...}
print(word_frequencies)
# -----------------------------------------------


# Task 4: Define a function that takes a string and an integer k, and returns a dictionary with
#   the token frequencies of only those tokens that occur more than k times in the string.

# Your code here:
# -----------------------------------------------
def token_counts(string: str, k: int = 1) -> dict:
    tokens = tokenize(string)
    return {word: tokens.count(word) for word in set(tokens) if tokens.count(word) > k}

# test:
text_hist = {'the': 2, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
# Note: token_counts(text) uses default k=1, so it returns only {'the': 2}
# To match text_hist (all words), we would need k=0. 
# However, strict adherence to the test logic:
assert all(text_hist[key] == value for key, value in token_counts(text, k=0).items())
# -----------------------------------------------


# [C] Sets & Dictionary comprehension: Mapping unique tokens to numbers and vice versa

# Task 5: Given a list of tokens from Exercise 1, construct two dictionaries:
#   `token_to_id`: a dictionary that maps each token to a unique integer ID.
#   `id_to_token`: a dictionary that maps each unique integer ID back to the original token.

# Your code here:
# -----------------------------------------------
# We use sorted(set(...)) here to create a clean vocabulary map from the token list
unique_tokens = sorted(set(tokenize(text)))
token_to_id = {word: i for i, word in enumerate(unique_tokens)}

# Expected output: {'brown': 0, 'dog': 1, 'fox': 2, 'jumps': 3, 'lazy': 4, 'over': 5, 'quick': 6, 'the': 7}
print(token_to_id)
# -----------------------------------------------


# Task 6: Define a dictionary that reverses the maping in `token2int`
#
# Your code here:
# -----------------------------------------------
id_to_token = {i: word for word, i in token_to_id.items()}

# tests: 
# test 1
assert id_to_token[token_to_id['dog']] == 'dog'
# test 2 (Assuming 'over' maps to 5 based on alphabetical sort)
assert token_to_id[id_to_token[4]] == 4
# test 3
assert all(id_to_token[token_to_id[key]]==key for key in token_to_id) and all(token_to_id[id_to_token[k]]==k for k in range(len(token_to_id)))
# -----------------------------------------------


# Task 7: Define a function that will take a list of strings ('documents'), determines all the
#   unique tokens across all documents, and returns two dictionaries: one (token2int) that maps 
#   each unique token to a unique integer, and a dictionary (int2token) that maps each integer
#   to the corresponding token in the first dictionary

# Your code here:
# -----------------------------------------------
def make_vocabulary_map(documents: list) -> tuple:
    all_text = " ".join(documents)
    unique_vocab = sorted(set(tokenize(all_text)))
    token2int = {word: i for i, word in enumerate(unique_vocab)}
    int2token = {i: word for i, word in enumerate(unique_vocab)}
    return token2int, int2token

# Test
t2i, i2t = make_vocabulary_map([text])
assert all(i2t[t2i[tok]] == tok for tok in t2i) # should be True
# -----------------------------------------------


# Task 8: Define a function that will take in a list of strings ('documents') and a vocabulary
#   dictionary token_to_id, that tokenizes each string in the list and returns a list with
#   each string converted into a list of token ID's.

# Your code here:
# -----------------------------------------------
def tokenize_and_encode(documents: list) -> list:
    # Get vocab maps
    token2int, int2token = make_vocabulary_map(documents)
    
    # Tokenize preserving order
    tokenized_documents = [tokenize(doc) for doc in documents]
    
    # Encode
    encoded_documents = [[token2int[token] for token in doc] for doc in tokenized_documents]
    return encoded_documents, token2int, int2token

# Test:
enc, t2i, i2t = tokenize_and_encode([text, 'What a luck we had today!'])
decoded = " | ".join([" ".join(i2t[i] for i in e) for e in enc])
print(decoded == 'the quick brown fox jumps over the lazy dog | what a luck we had today')
# -----------------------------------------------


# [D] Using a lambda expression to define functions: One line definition of a function

# Task 9: use a lambda function to implement the logistic function using the np.exp
#   function to work elementwise with numpy arrays

# Your code here:
# -----------------------------------------------
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Test:
np.all(sigmoid(np.log([1, 1/3, 1/7])) == np.array([1/2, 1/4, 1/8]))
# -----------------------------------------------


################  O P T I O N A L  ##############


# [E] Building an RNN layer

# Task 10: Translate this function into Python (by hand!)

# Your code here:
# -----------------------------------------------
def rnn_layer(w: np.array, list_of_sequences: list[np.array], sigma=sigmoid ) -> np.array:
    # NOTE: R matrices are Column-Major (Fortran order). Python is Row-Major (C order).
    # We must use order='F' to correctly replicate the R weights logic.
    W = w[:9].reshape(3,3, order='F')
    U = w[9:18].reshape(3,3, order='F')
    B = w[18:].reshape(1,3, order='F')

    nr_sequences = len(list_of_sequences)
    outputs = np.zeros(nr_sequences)

    for i in range(nr_sequences):
        X = list_of_sequences[i]
        # Initialize hidden state to 0
        a = np.zeros(X.shape[1])
        # Iterate over the time points
        for j in range(X.shape[0]):
            # The R reference code implements a Linear RNN (no sigma activation in loop)
            a = W @ X[j] + U @ a
        
        # store RNN output for i-th sequence
        # Use .item() to extract the scalar from the (1,) array result of B @ a
        outputs[i] = (B @ a).item()
        
    return outputs    
        
# Test
np.random.seed(10)
list_of_sequences_test = [np.random.normal(size=(5,3)) for _ in range(100)]
wstart = np.random.normal(size=(3*3 + 3*3 + 3)) 
o = rnn_layer(wstart, list_of_sequences_test)
print(o.shape == (100,) and o.mean().round(3) == 16.287 and o.std().astype(int) == 133)
# -----------------------------------------------


# [F] Defining a loss function

# Task 11: translate the above loss function into Python

# Your code here:
# -----------------------------------------------
def rnn_loss(w: np.array, list_of_sequences: list[np.array], y: np.array) -> np.float64:
    pred = rnn_layer(w, list_of_sequences)
    return np.sum((y - pred)**2)

# Test:
y_test = np.array([(X @ np.arange(1,4))[0] for X in list_of_sequences_test])
o_loss = rnn_loss(wstart, list_of_sequences_test, y_test)
print(o_loss.size == 1 and o_loss.round(3) == 17794.733)
# -----------------------------------------------


# [G] Fitting the RNN with minimize for the scipy.optmize module

data = macrodata.load_pandas().data
X = np.hstack([np.ones((len(data),1)), data[['cpi','unemp']].values]) # Features: CPI and unemployment
y = data['infl'].values # Target: inflation

seq_len = 7 

data_pairs = [(X[i:i+seq_len], y[i+seq_len]) for i in range(len(X)-seq_len)]

list_of_sequences, yy = list(zip(*data_pairs))

# fit the RNN (this may take a minute)
fit = minimize(rnn_loss, wstart, args=(list_of_sequences, yy), method='BFGS')
print(fit)

# Evaluate
pred = rnn_layer(fit['x'], list_of_sequences)
print("Correlation RNN:", np.corrcoef(pred,yy)[0, 1])

# Compare with Linear Regression
Z = X[:len(yy)] 
linreg_coefs = np.linalg.lstsq(Z, yy, rcond=None)[0] 
linreg_pred = Z @ linreg_coefs
print("Correlation LinReg:", np.corrcoef(linreg_pred, yy)[0, 1])

# Plot
plt.figure(figsize=(10, 5))
plt.plot(yy, label='Truth', alpha=0.7)
plt.plot(pred, label='RNN', alpha=0.7)
plt.plot(linreg_pred, label='LinReg', alpha=0.7, linestyle='--')
plt.legend()
plt.show()