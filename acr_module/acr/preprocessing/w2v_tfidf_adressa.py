import argparse
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from ..utils import serialize, deserialize
from .tokenization import tokenize_articles, nan_to_str, convert_tokens_to_int, get_words_freq

from nltk.tokenize import word_tokenize


def create_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input_articles_csv_path', default='',
            help='Input path of the news CSV file.')

    parser.add_argument(
            '--input_word_embeddings_vocab', default='',
            help='Input path for a pickle with word embeddings and its vocab dict.')

    parser.add_argument(
            '--output_article_content_embeddings', default='',
            help='')
    return parser

VECTOR_SIZE = 250

def load_input_csv(path):
    news_df = pd.read_csv(path, encoding = 'utf-8' 
                            #,nrows=1000
                            )
    #Making sure articles are sorted by there encoded id
    news_df.sort_values('id_encoded', inplace=True)
    return news_df


def tokenize_norwegian_article(text, first_sentences=12, max_words_length=1000):
    #Removing pipes for correct sentence tokenization
    text = text.replace('|', '.')
    words_tokenized = []
    sent_count = 0
    for sentence in nltk.tokenize.sent_tokenize(text, language='norwegian'):        
        sent_tokenized = nltk.tokenize.word_tokenize(sentence, language='norwegian')
        if len(sent_tokenized) >= 3 and sent_tokenized[-1] in ['.', '!', '?', ';'] and \
           sent_tokenized != ['Saken', 'oppdateres', '.']:                
            sent_count += 1
            words_tokenized.extend(sent_tokenized)        
            if sent_count == first_sentences:
                break
    return words_tokenized[:max_words_length]


def export_article_content_embeddings(content_article_embeddings, output_article_content_embeddings):
    output_path = output_article_content_embeddings
    print('Exporting ACR Label Encoders, Article metadata and embeddings to {}'.format(output_path))
    #to_serialize = (acr_label_encoders, articles_metadata_df, content_article_embeddings)
    to_serialize = content_article_embeddings
    serialize(output_path, to_serialize)


def main():
    parser = create_args_parser()
    args = parser.parse_args()

    print('Loading news article CSV: {}'.format(args.input_articles_csv_path))
    news_df = load_input_csv(args.input_articles_csv_path)
    print('N. docs: {}'.format(len(news_df)))

    
    assert news_df['id_encoded'].values[0] == 1
    assert news_df['id_encoded'].max() == len(news_df)
    assert len(news_df[pd.isnull(news_df['id_encoded'])]) == 0   
    

    print('Loading word embeddings')
    (word_vocab, word_embeddings_matrix) = deserialize(args.input_word_embeddings_vocab)
    print('word_embeddings_matrix', word_embeddings_matrix.shape)



    print('Tokenizing articles...')
    tokenized_articles = tokenize_articles(news_df['text_highlights'].values, tokenization_fn=tokenize_norwegian_article)
    

    print('TF-IDF...')
    vectorizer = TfidfVectorizer(analyzer='word', 
                             tokenizer=lambda x: x,
                             preprocessor=lambda x: x,
                             token_pattern=None,
                             stop_words=None, 
                             ngram_range=(1, 1), max_df=0.4, 
                             min_df=2, max_features=50000, 
                             norm='l2', use_idf=True, 
                             smooth_idf=True, sublinear_tf=False)

    vectorized_contents = vectorizer.fit_transform(tokenized_articles)
    print('vectorized_content.shape={}'.format(vectorized_contents.shape))
    feature_names = np.array(vectorizer.get_feature_names())

    print('Averaging word embeddings weighted by TF-IDF score')
    invalid_embeddings = []
    content_embeddings = []
    for article_idx, vect_content in tqdm(enumerate(vectorized_contents)):
        word_idxs = vect_content.nonzero()[1]
        word_weights = vect_content[0,word_idxs].todense().tolist()[0]
        word_names = feature_names[word_idxs]

        words = []
        weights = []
        for word, weight in zip(word_names, word_weights):
            if word in word_vocab:
                words.append(word_embeddings_matrix[word_vocab[word]] * weight)
                weights.append(weight)
            #else:
            #    print('Word not found: {}'.format(word))

        if len(words) > 0:
            avg_word_embedding = np.vstack(words).sum(axis=0) / sum(weights)
        else:
            print('No valid words for article_idx: {}'.format(article_idx))
            invalid_embeddings.append(article_idx)
            avg_word_embedding = word_embeddings_matrix[1] #Using the <UNK> token


        content_embeddings.append(avg_word_embedding)

    print('Total invalid embeddings: {}'.format(len(invalid_embeddings)))
    print('Invalid embeddings: {}'.format(invalid_embeddings))

    print('Concatenating article content embeddings')
    content_embeddings_concat = np.vstack(content_embeddings)
    print('content_embeddings_concat', content_embeddings_concat.shape)

    #As word embeddings for Adressa have 100-dim, concatenating 3 times to get a "300-dim" embedding, to be able to reduce to 250
    content_embeddings_hconcat = np.hstack([content_embeddings_concat, content_embeddings_concat, content_embeddings_concat])
    print('content_embeddings_hconcat', content_embeddings_hconcat.shape)

    pca = PCA(n_components=250)
    article_content_embeddings = pca.fit_transform(content_embeddings_hconcat)

    print('article_content_embeddings.shape={}'.format(article_content_embeddings.shape))

    #Checking if content articles embedding size correspond to the last article_id
    assert article_content_embeddings.shape[0] == news_df['id_encoded'].tail(1).values[0]



    print('Concatenating article content embeddings, making sure that they are sorted by the encoded article id') 
    embedding_for_padding_article = np.mean(article_content_embeddings, axis=0)
    content_article_embeddings_with_padding = np.vstack([embedding_for_padding_article, article_content_embeddings])
    del article_content_embeddings

    

    print('Exporting article content embeddings')
    del news_df
    export_article_content_embeddings(content_article_embeddings_with_padding, args.output_article_content_embeddings)

    #Ps: To experiment with these Content embeddings, it is necessary to deserialize the "acr_articles_metadata_embeddings.pickle"
    #trained by the ACR module, which is a tuple like (acr_label_encoders, articles_metadata_df, content_article_embeddings),
    #substitute only the content_article_embedding instance by content embedddings produced by this script
    # and serialize it again for further usage by NAR module


if __name__ == '__main__':
    main()


'''
DATA_DIR=/media/data/projects/personal/doutorado/adressa_news/data_transformed && \
python3 -m acr.preprocessing.w2v_tfidf_adressa \
    --input_articles_csv_path ${DATA_DIR}/articles_tfrecords_v4_first_12_sent.csv \
    --input_word_embeddings_vocab ${DATA_DIR}/pickles_v4/acr_word_vocab_embeddings_v4.pickle \
    --output_article_content_embeddings ${DATA_DIR}/pickles_v4/article_content_embeddings_v4_lsa_w2v_tfidf.pickle

'''