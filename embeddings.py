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


import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os

def get_qwen_embeddings(train_texts, val_texts, model_name="Qwen/Qwen2.5-1.5B", batch_size=128, cache_dir="./embedding_cache", save=True):
    if save:
        os.makedirs(cache_dir, exist_ok=True)
        train_path = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_train.npy")
        val_path = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_val.npy")

        if os.path.exists(train_path) and os.path.exists(val_path):
            print(f"Loading saved embeddings from {cache_dir}")
            return np.load(train_path), np.load(val_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()

    def encode(texts):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).half()
            embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    train_emb = encode(list(train_texts))
    val_emb = encode(list(val_texts))

    if save:
        np.save(train_path, train_emb)
        np.save(val_path, val_emb)
        print(f"Saved embeddings to {cache_dir}")

    del model
    torch.cuda.empty_cache()

    return train_emb, val_emb