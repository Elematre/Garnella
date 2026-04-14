from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

##### BASELINES ######

# BAG-OF-WORDS REPRESENTATION
# 1) Building the vocabulary (fit step): CountVectorizer scans all sentences in train_df["sentence"] and builds a vocabulary of up to 10,000 of the most frequent tokens. With ngram_range=(1, 2), those tokens include both single words (unigrams) and consecutive word pairs (bigrams)
# 2) Transforming to count vectors (transform step): Each sentence is then represented as a sparse vector of length 10,000, where each element is the count of how many times that vocabulary token appears in the sentence. Most entries are 0

# Important: Fit ONLY on training data
def get_bagOfWords_embeddings(X_train, X_test):
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

def get_tfidf_embeddings(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test


### pretrained word embeddings, TODO maybe don't take in X_train and X_test as arguments?

def get_gloVe_embeddings(X_train, X_test):  
    # TODO: implement GloVe embeddings
    return None, None