from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sentence_transformers import SentenceTransformer

##### BASELINES ######

# Important: Fit ONLY on training data
def get_bagOfWords_embeddings(X_train, X_test):
    # ngram_range=(1, 2): tokens include single words (unigrams) and consecutive word pairs (bigrams)
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
    # 1) build vocabulary of up to 10'000 most frequent tokens
    X_train = vectorizer.fit_transform(X_train)
    # 2) transform sentences to count vectors
    # Each sentence is represented as a sparse vector of length 10,000, where each element is the count of how many times that vocabulary token appears in the sentence. Most entries are 0
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

# BoW but on characters -> more robust to spelling errors
def get_char_ngram_embeddings(X_train, X_test):
    vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=10000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

def get_tfidf_embeddings(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test


### pretrained word embeddings ###

# TODO: maybe use multilingual pre-trained embeddings
def get_multilingual_embeddings(X_train, X_val):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    X_train_emb = model.encode(list(X_train))
    X_val_emb   = model.encode(list(X_val))
    return X_train_emb, X_val_emb



# NOTE: tried GloVe but realised it is only for English,so doesn't work for our mix here
# def load_glove(path="glove.6B.100d.txt"):
#     embeddings = {}
#     with open(path, "r", encoding="utf-8") as file:
#         for line in file:
#             parts = line.split()
#             word = parts[0]
#             vector = np.array(parts[1:], dtype=np.float32)
#             embeddings[word] = vector
#     return embeddings

# def embed_sentence(sentence, glove_dict, dim=100):
#     words = sentence.lower().split()
#     vectors = [glove_dict[w] for w in words if w in glove_dict]
#     if len(vectors) == 0:
#         return np.zeros(dim)  # fallback for unknown sentences
#     # average the word vectors to get a single vector representation for the sentence
#     return np.mean(vectors, axis=0)

# def get_glove_embeddings(X_train, X_val, path="glove.6B.100d.txt", dim=100):
#     glove_dict = load_glove(path)
#     X_train_emb = np.array([embed_sentence(s, glove_dict, dim) for s in X_train])
#     X_val_emb   = np.array([embed_sentence(s, glove_dict, dim) for s in X_val])
#     return X_train_emb, X_val_emb