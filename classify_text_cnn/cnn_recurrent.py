## Please read this paper before going through the code: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552
import gensim
import numpy as np
import string

from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model

word2vec = gensim.models.Word2Vec.load("word2vec.gensim")

# In order to represent unseen words and the null token; added an additional row of zeros to the embeddings matrix.
embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype = "float32")
embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

maximum_tokens = word2vec.syn0.shape[0]
embedding_dim = word2vec.syn0.shape[1]
hidden_dim_1 = 200
hidden_dim_2 = 100
NUM_CLASSES = 10

document = Input(shape = (None, ), dtype = "int32")
left_context = Input(shape = (None, ), dtype = "int32")
right_context = Input(shape = (None, ), dtype = "int32")

embedder = Embedding(maximum_tokens + 1, embedding_dim, weights = [embeddings], trainable = False)
document_embedding = embedder(document)
left_embedding = embedder(left_context)
right_embedding = embedder(right_context)


forward = LSTM(hidden_dim_1, return_sequences = True)(left_embedding) # Using LSTM RNNs instead of vanilla RNNs. As detailed in equation (1) of the paper.
backward = LSTM(hidden_dim_1, return_sequences = True, go_backwards = True)(right_embedding) # equation (2) of the paper.
together = concatenate([forward, document_embedding, backward], axis = 2) # Equation (3).

semantic = TimeDistributed(Dense(hidden_dim_2, activation = "tanh"))(together) # Equation (4) of the paper using tanh activation .

pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(semantic) # See equation (5).

output = Dense(NUM_CLASSES, input_dim = hidden_dim_2, activation = "softmax")(pool_rnn) # Equations (6) &(7) of the paper.

model = Model(inputs = [document, left_context, right_context], outputs = output)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])

text = "Insert some example text."
text = text.strip().lower().translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
tokens = text.split()
tokens = [word2vec.vocab[token].index if token in word2vec.vocab else maximum_tokens for token in tokens]

doc_as_array = np.array([tokens])

left_context_as_array = np.array([[maximum_tokens] + tokens[:-1]]) # Shifting the document to the right to obtain the left-side contexts.

right_context_as_array = np.array([tokens[1:] + [maximum_tokens]]) # Shifting the document to the left to obtain the right-side contexts.

target = np.array([NUM_CLASSES * [0]])
target[0][3] = 1

history = model.fit([doc_as_array, left_context_as_array, right_context_as_array], target, epochs = 1, verbose = 0)
loss = history.history["loss"][0]
