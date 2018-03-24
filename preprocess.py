import re

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import rcv1_constants as dataset


def preprocess(text, vocab):
    text = clean_text(text)
    data_tfidf, tfidf_vectorizer = tfidf_vectorize([text], vocab)
    data_tfidf = data_tfidf.toarray()  # convert sparse matrix to array
    data_word2ind = generate_word2ind([text], vocab, tfidf_vectorizer)
    return data_tfidf, data_word2ind


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9(),!?'$]", " ", text)
    text = re.sub(r"(\d+)", " NUM ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


def tfidf_vectorize(documents, vocab):
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)
    tfidf_vectorizer.fit(documents)
    data_tfidf = tfidf_vectorizer.transform(documents)
    assert len(vocab) == data_tfidf.shape[1]
    return data_tfidf, tfidf_vectorizer


def generate_word2ind(documents, vocab, tfidf_vectorizer):
    # Parameters for generating word2ind
    maxlen = dataset.SEQ_LEN
    padding = "post"
    truncating = "post"

    # Add "<UNK>" to vocabulary (for padding) and create a reverse vocabulary lookup
    if vocab[-1] != "<UNK>":
        vocab = vocab + ["<UNK>"]
    reverse_vocab = {w: i for i, w in enumerate(vocab)}

    # Tokenize all the documents using the TfidfVectorizer's analyzer
    analyzer = tfidf_vectorizer.build_analyzer()
    tokenized_docs = np.array([analyzer(doc) for doc in documents])

    # Transform documents from words to indexes using vocabulary
    sequences = np.array([[reverse_vocab[w] for w in tokens if w in reverse_vocab]
                          for tokens in tokenized_docs])

    # Truncate or pad sequences to match maxlen (adapted from tflearn.data_utils.pad_sequences)
    lengths = [len(s) for s in sequences]
    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = np.ones((num_samples, maxlen), np.int64) * (len(vocab) - 1)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]

        if padding == "post":
            x[idx, :len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc):] = trunc

    return x
