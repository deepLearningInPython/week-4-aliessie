import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.

# [A] List Comprehensions and String Manipulation: Tokenization
#     Objective: Practice list comprehensions and basic string operations: split a sentence 
#                into individual words and use list comprehensions to make the code cleaner 
#                and more readable.

# List comprehension provides a concise way to create lists by embedding a for-loop inside 
# square brackets.
# Syntax: [expression for item in iterable if condition] (condition is optional).
# Example: squares = [x**2 for x in range(10) if x % 2 == 0]

# Large language models work with "tokens," which are the basic units of text (often words or subwords). 
# Tokenization is the process of breaking down sentences into these tokens. In this exercise, you’ll 
# create a simple tokenizer to split a sentence into words and remove punctuation.



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
#   punctuation, and then converts each token to lowercase. The function should returns unique 
#   words in alphabetical order.

# Your code here:
# -----------------------------------------------
def tokenize(string: str) -> list:
    str_list = [word.strip('!.,?').lower() for word in string.split(" ")]
    return sorted(set(str_list))

# -----------------------------------------------

# [B] Dictionary Comprehensions: Frequency Count of Tokens
#     Objective: Practice dictionary comprehensions for token frequency counts.

# Dictionary comprehension is a concise way to create dictionaries using a for-loop inside curly braces.
# Syntax: {key: value for item in iterable if condition} (condition is optional).
# Example: char_count = {char: ord(char) for char in "hello" if char != 'e'}

# Once tokens are extracted, a common task in NLP is to count how often each word appears. 
# This is called calculating the frequency of tokens, and it’s useful because words that appear 
# frequently might have different importance compared to rare words. In this exercise, you’ll 
# create a dictionary where each word is a key and its frequency (count) is the value.



# Task 3: Using the tokens list from the previous exercise, create a dictionary comprehension 
#   that counts the frequency of each word.

# Using the list of tokens from Exercise 1, count the frequency of each word within one 
# dictionary comprehension

# Your code here:
# -----------------------------------------------
word_frequencies = {word: tokenize(text).count(word) for word in tokenize(text)}

# Expected output example: {'the': 2, 'quick': 1, ...}
print(word_frequencies)

# Modify the comprehension to include only words that appear more than once.
# -----------------------------------------------



# Task 4: Define a function that takes a string and an integer k, and returns a dictionary with
#   the token frequencies of only those tokens that occur more than k times in the string.

# Your code here:
# -----------------------------------------------
def token_counts(string: str, k: int = 1) -> dict:
    return {word: tokenize(string).count(word) for word in tokenize(string) if tokenize(string).count(word) > k}

# test:
text_hist = {'the': 2, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
all(text_hist[key] == value for key, value in token_counts(text).items())
# -----------------------------------------------




# [C] Sets & Dictionary comprehension: Mapping unique tokens to numbers and vice versa
#   Objective: Practice dictionary comprehensions and create mappings from tokens to unique 
#              numerical IDs and back.

# Once tokens are created, they often need to be converted to numerical representations 
# for use in models. Two essential mappings are:
#
# Token to ID: Maps each token to a unique number.
# ID to Token: Maps each unique number back to its corresponding token.
#
# These mappings are necessary for transforming text data into numerical data that models can 
# process. In this exercise, you’ll use dictionary comprehensions to create these mappings.



# Task 5: Given a list of tokens from Exercise 1, construct two dictionaries:
#   `token_to_id`: a dictionary that maps each token to a unique integer ID.
#   `id_to_token`: a dictionary that maps each unique integer ID back to the original token.

# Your code here:
# -----------------------------------------------
token_to_id = {word: i for i, word in enumerate(tokenize(text))}

# Expected output: {'dog': 0, 'quick': 1, 'fox': 2, 'the': 3, 'over': 4, 'lazy': 5, 'brown': 6, 'jumps': 7}
print(token_to_id)
# -----------------------------------------------



# Task 6: Define a dictionary that reverses the maping in `token2int`
#
# Your code here:
# -----------------------------------------------
id_to_token = {i: word for i, word in enumerate(tokenize(text))}

# tests: 
# test 1
assert id_to_token[token_to_id['dog']] == 'dog'
# test 2
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
    token2int = {word: i for i, word in enumerate(tokenize(" ".join(documents)))}
    int2token = {i: word for i, word in enumerate(tokenize(" ".join(documents)))}
    return token2int, int2token

# Test
t2i, i2t = make_vocabulary_map([text])
all(i2t[t2i[tok]] == tok for tok in t2i) # should be True
# -----------------------------------------------



# Task 8: Define a function that will take in a list of strings ('documents') and a vocabulary
#   dictionary token_to_id, that tokenizes each string in the list and returns a list with
#   each string converted into a list of token ID's. For example:
#   ['Good day!', 'What a day'] -> [[0, 1], [2, 3, 1]]
#   The function should return three values: the list of encoded sentences, the token_to_id,
#   and the id_to_token dictionaries.

# Your code here:
# -----------------------------------------------
def tokenize_and_encode(documents: list) -> list:
    token2int, int2token = make_vocabulary_map(documents)
    tokenized_documents = [tokenize(doc) for doc in documents]
    encoded_documents = [[token2int[token] for token in doc] for doc in tokenized_documents]
    return encoded_documents, token2int, int2token

# Test:
enc, t2i, i2t = tokenize_and_encode([text, 'What a luck we had today!'])
" | ".join([" ".join(i2t[i] for i in e) for e in enc]) == 'the quick brown fox jumps over the lazy dog | what a luck we had today'
# -----------------------------------------------



# In the following set of exercises you're going to implement an RNN from scratch. You'll also
# fit it to an existing time series.


# [D] Using a lambda expression to define functions: One line definition of a function
# Objective: practicing to work with lambda functions

# You'll implement a RNN with the logistic (sigmoid) activation function for
# the nodes. We need to implement this function first.



# Task 9: use a lambda function to implement the logistic function using the np.exp
#   function to work elementwise with numpy arrays

# Your code here:
# -----------------------------------------------
sigmoid = _ # Your code

# Test:
np.all(sigmoid(np.log([1, 1/3, 1/7])) == np.array([1/2, 1/4, 1/8]))
# -----------------------------------------------


################  O P T I O N A L  ##############


# [E] Building an RNN layer
# Objective: Gaining a computational understanding of an RNN

# The equations of an RNN are
#
# a[t] = W x[t] + U a[t-1]
# o[t] = B a[t]
#
# All the multiplications here are matrix-vector multiplications (i.e., W, U and B
# are matrices, and x[t] and a[t] are vectors). 
# To implement an RNN layer, these equations need to be implemented. The values
# that define the matrices need to be passed to the layer. 
# 

# And implementation in R may look as follows:
#
# rnn_layer = function(w, list_of_sequences, sigma=plogis) {
#   # 1. Setup
# 	W = matrix(w[1:9],3,3)
# 	U = matrix(w[1:9 + 9], 3, 3)
# 	B = matrix(w[1:3+9+9],1,3)
#
# 	nr_sequences = length(list_of_sequences)
# 	outputs = rep(NA, nr_sequences)
#
#   # 2. Iterate over sequences
# 	for (i in 1:nr_sequences) {
# 		# get i-th sequence
# 		X = list_of_sequences[[i]]
# 		# initialize hidden state to 0
# 		a = 0 * X[1,]
# 		# Iterate over the time points
# 		for (j in 1:nrow(X)) {
# 			a = W %*% X[j,] + U %*% a
# 		}
# 		# store RNN output for i-th sequence
# 		outputs[i] = B %*% a
# 	}
# 	outputs
# }

# In this implementation 
# w:                 is a single vector containing the flattened weights for the matrices  W ,  U , and  B .
# list_of_sequences: is a list where each element is a sequence represented as a matrix ( X ), 
#                    where rows are time steps and columns are input features.
# sigma:             An optional activation function (default is the sigmoid function plogis).
# 1. Setup: Splits the vector w into three matrices ( W, U, B ).
#       and determines the number of sequences to process.
# 2. Iterate Over Sequences: For each sequence in list_of_sequences 
#      the sequence matrix  X is extracted and processed by
#      • first initializing the hidden state  a  to zero (vector of the same size as a row of  X).
#      • then, for each time step (each row in  X ) the hidden state  a[t] is updated using the RNN 
#        formula:  a[t] = W  x[t] + U a[t-1].
# 4. Compute Output: After processing all time steps of the sequence, the output value is
#      computed using final hidden state using the the equation o = B a[T] and is
#      stored as the predicted output value for this sequence.
# The return value is the vector of RNN output values (one value for each sequence in list_of_sequences).



# Task 10: Translate this function into Python (by hand!)

# Your code here:
# -----------------------------------------------
def rnn_layer(w: np.array, list_of_sequences: list[np.array], sigma=sigmoid ) -> np.array:
    pass # Your code

# Test
np.random.seed(10)
list_of_sequences = [np.random.normal(size=(5,3)) for _ in range(100)]
wstart = np.random.normal(size=(3*3 + 3*3 + 3)) 
o = rnn_layer(wstart, list_of_sequences)
o.shape == (100,) and o.mean().round(3) == 16.287 and o.std().astype(int) == 133
# -----------------------------------------------




# [F] Defining a loss function
# Objective: define the least squares loss function suitable for minimizing with scipy.optimize.minimize

# We want to predict the target value y from each of the sequences using our RNN. We'll do
# that with the minimize function from the scipy.optimize module. But we'll need to implement
# a loss function.

# In R the loss function may look as follows:
#
# rnn_loss = function(w, list_of_sequences, y) {
# 	pred = rnn_layer(w, list_of_sequences)
# 	sum((y-pred)^2)
# }




# Task 11: translate the above loss function into Python

# Your code here:
# -----------------------------------------------
# def rnn_loss(w: np.array, w, list_of_sequences: list[np.array], y: np.array) -> np.float64:
#     pass # Your code

# Test:
y = np.array([(X @ np.arange(1,4))[0] for X in list_of_sequences])
o = rnn_loss(wstart, list_of_sequences, y)
o.size == 1 and o.round(3) == 17794.733
# -----------------------------------------------




# [G] Fitting the RNN with minimize for the scipy.optmize module
# Objective: fit your RNN on real data

# The data that we will fit is a macroeconomics data set. We'll try to predict inflation ('infl')
# from the consumer price index ('cpi') and unemployment rate ('unemp').
# First, load the data set:
from statsmodels.datasets import macrodata

data = macrodata.load_pandas().data
X = np.hstack([np.ones((len(data),1)), data[['cpi','unemp']].values]) # Features: CPI and unemployment
y = data['infl'].values # Target: inflation

# Next we want to prepare a dataset for training sequence-based models like RNNs. We create 
# input-output pairs where each input is a sequence of seq_len time steps from X, and the output 
# is the corresponding target value y at the next time step after the sequence.

seq_len = 7 # Define the length of each input sequence (we choose 7 consecutive time steps).

# Create a list of tuples:
data_pairs = [(X[i:i+seq_len], y[i+seq_len]) for i in range(len(X)-seq_len)]
# - First element: a slice of `X` of length `seq_len` (the input sequence).
# - Second element: the target value `y` corresponding to the step after the sequence.
# Example: If seq_len=4, for i=0, pair is (X[0:4], y[4]).

# We need the input sequences and target values in a separate list. A trick to do this is this:

list_of_sequences, yy = list(zip(*data_pairs))

# Here, the zip(*...) is used for transposing a list of tuples. It splits the tuple pairs into 
# two separate lists:
# First list: all input sequences (X[i:i+seq_len])
# Second list: all target values (y[i+seq_len])
# The * operator in Python unpacks the list elements into separate arguments for the zip() 
# function. E.g., func(*[2,4,5]) is the same as func(2,4,5). 

# Now we are ready to fit the RNN to the data set. We need to load the optimization routine 
# 'minimize' from the scipy.optimize module

from scipy.optimize import minimize

# fit the RNN (this may take a minute)
fit = minimize(rnn_loss, wstart, args=(list_of_sequences, yy), method='BFGS')
print(fit)

# The 'success' component in fit may be false, and this is due to a loss of computational 
# precision. For now we'll just settle for the weights it has found so far. 

# To evaluate the fit we can compute the correlation between the values predicted by the
# RNN and the true values
pred = rnn_layer(fit['x'], list_of_sequences)
np.corrcoef(pred,yy)

# How good is this? To gage the performance of the RNN we'll compare it to a linear 
# regression with the same data
Z = X[:len(yy)] # features corresponding to elements in yy at the previous time step
linreg_coefs = np.linalg.lstsq(Z, yy, rcond=None)[0] # rcond=None suppresses warning message
linreg_pred = Z @ linreg_coefs
np.corrcoef(linreg_pred, yy)

# The correlation of the RNN predicted values is substantially higher! But it also has
# many more parameters, and so is more flexible. 

# To visualize the difference in performance we plot the true values and predicted values
import matplotlib.pyplot as plt

plt.plot(yy)
plt.plot(pred)
plt.plot(linreg_pred)
plt.legend(['Truth','RNN','LinReg'])


