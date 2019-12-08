import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from ..utils import serialize
from ..acr_commons import PAD_TOKEN, UNK_TOKEN

def load_word_embeddings(path, binary=True):
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=binary)
    return w2v_model

def process_word_embedding_for_corpus_vocab(w2v_model, words_freq, 
                                            keep_most_frequent_words=100000):
    print('Tokens vocab. from articles: {}'.format(len(words_freq)))    
    most_freq_words = set(list(map(lambda x: x[0], words_freq.most_common(keep_most_frequent_words))))
    print('Most common tokens vocab. from articles: {}'.format(len(most_freq_words)))

    RESERVED_TOKENS_IN_VOCAB=2

    embedding_size = w2v_model.vector_size
    new_embeddings_list = []
    new_vocab = {}
    last_token_id = RESERVED_TOKENS_IN_VOCAB

    w2v_vocab = set(w2v_model.wv.index2word)
    for word in most_freq_words:        
        if word in w2v_vocab:    
            new_vocab[word] = last_token_id
            last_token_id += 1
            new_embeddings_list.append(w2v_model[word])
            

    #Inserting the 2 reserved tokens
    new_vocab[PAD_TOKEN] = 0
    new_vocab[UNK_TOKEN] = 1

    np.random.seed(10)
    unk_vector = np.random.uniform(low=-0.04, high=0.04, size=embedding_size)
    pad_vector = np.random.uniform(low=-0.04, high=0.04, size=embedding_size)

    new_embeddings_matrix = np.vstack([unk_vector, pad_vector] + new_embeddings_list)

    print('Most common tokens with word embeddings: {}'.format(new_embeddings_matrix.shape[0]))

    return new_vocab, new_embeddings_matrix


def save_word_vocab_embeddings(output_path, word_vocab, word_embeddings_matrix):
    to_serialize = (word_vocab, word_embeddings_matrix)
    serialize(output_path, to_serialize)