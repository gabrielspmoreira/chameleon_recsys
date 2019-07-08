import argparse
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder

from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from ..utils import serialize, deserialize
from .tokenization import tokenize_articles, nan_to_str, convert_tokens_to_int, get_words_freq

from nltk.tokenize import word_tokenize


def create_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input_articles_csv_path', default='',
            help='Input path of the news CSV file.')

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
   

    print('Tokenizing articles...')
    

    print('LSA = TF-IDF + SVD...')
    #https://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/
    vectorizer = TfidfVectorizer(analyzer='word', 
                             tokenizer=lambda x: x,
                             preprocessor=lambda x: x,
                             token_pattern=None,
                             stop_words=None, 
                             ngram_range=(1, 3), max_df=0.4, 
                             min_df=2, max_features=50000, 
                             norm='l2', use_idf=True, 
                             smooth_idf=True, sublinear_tf=False)


    svd = TruncatedSVD(n_components=VECTOR_SIZE)

    lsa = make_pipeline(vectorizer, svd, Normalizer(copy=False))

    reduced_content = lsa.fit_transform(tokenized_articles)

    print('Concatenating article content embeddings, making sure that they are sorted by the encoded article id')
    article_content_embeddings = np.vstack(reduced_content)
    #Checking if content articles embedding size correspond to the last article_id
    assert article_content_embeddings.shape[0] == news_df['id_encoded'].tail(1).values[0]
        
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
Example command to run:

DATA_DIR="[REPLACE BY THE ADRESSA ARTICLES DATASET PATH]" && \
python3 -m acr.preprocessing.lsa_adressa \
    --input_articles_csv_path ${DATA_DIR}/adressa_articles_converted_to_csv_by_preproc.csv \
    --output_article_content_embeddings ${DATA_DIR}/pickles/article_content_embeddings_lsa_trigram.pickle

'''