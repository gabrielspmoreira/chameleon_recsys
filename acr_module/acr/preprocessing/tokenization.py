import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords

from ..acr_commons import UNK_TOKEN


def nan_to_str(value):
    return '' if type(value) == float else value  

def get_words_freq(tokenized_articles):
    words_freq = FreqDist([word for article in tokenized_articles for word in article])
    return words_freq    

def tokenize_text(text, clean_str_fn, lower_first_word_sentence=False):
    text = clean_str_fn(text)
    tokenized_text = []
    new_sentence = False
    for word in word_tokenize(text):
        if word in ['.', '?', '!']:
            new_sentence = True
        else:
            if lower_first_word_sentence and new_sentence:
                word = word.lower()
                new_sentence = False

        tokenized_text.append(word)

    return tokenized_text  

def tokenize_articles(articles, tokenization_fn=None, clean_str_fn=lambda x: x):
    if tokenization_fn == None:
        tokenized_articles = [tokenize_text(text, clean_str_fn) for text in articles]
    else:
        tokenized_articles = [tokenization_fn(text) for text in articles]
    return tokenized_articles

def print_vocab_tokens_stats(tokenized_int_texts, texts_lengths, word_vocab):
    print('# tokens by article stats - Mean: {:.1f}, Median: {:.1f}, Max: {:.1f}'.format(
          np.mean(texts_lengths), np.median(texts_lengths), np.max(texts_lengths))
         )

    perc_words_found_vocab = (sum([len(list(filter(lambda word: word != word_vocab[UNK_TOKEN], doc))) for doc in tokenized_int_texts]) / \
                              float(sum(texts_lengths))) * 100
    print('{:.2f}%  tokens were found in vocabulary.'.format(perc_words_found_vocab))

def convert_tokens_to_int(tokenized_articles, word_vocab):
    
    def token_to_int(token):
        return word_vocab[token] if token in word_vocab else word_vocab[UNK_TOKEN] 

    texts_int = list([np.array([token_to_int(token) for token in article]) for article in tokenized_articles])
    texts_lengths = np.array([len(doc) for doc in texts_int])
    print_vocab_tokens_stats(texts_int, texts_lengths, word_vocab)

    return texts_int, texts_lengths