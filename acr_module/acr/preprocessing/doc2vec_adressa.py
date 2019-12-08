import argparse
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder


from ..utils import serialize
from .tokenization import tokenize_articles, nan_to_str, convert_tokens_to_int, get_words_freq

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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

def load_input_csv(path):
    news_df = pd.read_csv(path, encoding = 'utf-8' 
                            #,nrows=1000
                            )
    #Making sure articles are sorted by there encoded id
    news_df.sort_values('id_encoded', inplace=True)
    return news_df

'''
def process_cat_features(dataframe):
    article_id_encoder = LabelEncoder()
    dataframe['id_encoded'] = article_id_encoder.fit_transform(dataframe['id'])

    #category_id_encoder = LabelEncoder()
    #dataframe['categoryid_encoded'] = category_id_encoder.fit_transform(dataframe['categoryid'])

    #domainid_encoder = LabelEncoder()
    #dataframe['domainid_encoded'] = domainid_encoder.fit_transform(dataframe['domainid'])


    return article_id_encoder#, category_id_encoder, domainid_encoder


def save_article_cat_encoders(output_path, article_id_encoder, category_id_encoder, domainid_encoder):
    to_serialize = {'article_id': article_id_encoder, 
                    'category_id': category_id_encoder, 
                    'publisher_id': domainid_encoder}
    serialize(output_path, to_serialize)
'''

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

    '''
    print('Encoding categorical features')
    article_id_encoder, category_id_encoder, domainid_encoder = process_cat_features(news_df)
    print('Exporting LabelEncoders of categorical features: {}'.format(args.output_label_encoders))
    save_article_cat_encoders(args.output_label_encoders, 
                              article_id_encoder, 
                              category_id_encoder, 
                              domainid_encoder)
    '''

    print('Tokenizing articles...')
    tokenized_articles = tokenize_articles(news_df['text_highlights'].values, tokenization_fn=tokenize_norwegian_article)

    #print('Computing word frequencies...')
    #words_freq = get_words_freq(tokenized_articles)
    #print('Corpus vocabulary size: {}'.format(len(words_freq)))

    print('Processing documents...')
    tagged_data = [TaggedDocument(words=w, tags=[i]) for i, w in enumerate(tokenized_articles)]    


    print('Training doc2vec')
    max_epochs = 30
    vec_size = 250
    alpha = 0.025
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=alpha,   
                    window=5,
                    negative=5,
                    min_count=2,                                     
                    max_vocab_size=100000,
                    dm = 1,
                    dm_mean=1,
                    workers=6)
      
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=1) #model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    del tokenized_articles


    #print('Encoding categorical features')
    #article_id_encoder = process_cat_features(news_df)

    print('Concatenating article content embeddings, making sure that they are sorted by the encoded article id')
    article_content_embeddings = np.vstack([model.docvecs[i-1] for i in news_df['id_encoded'].values])    
    embedding_for_padding_article = np.mean(article_content_embeddings, axis=0)
    content_article_embeddings_with_padding = np.vstack([embedding_for_padding_article, article_content_embeddings])
    del article_content_embeddings

    #Checking if content articles embedding size correspond to the last article_id
    assert content_article_embeddings_with_padding.shape[0] == news_df['id_encoded'].tail(1).values[0]+1

    print('Exporting article content embeddings')
    del news_df
    export_article_content_embeddings(content_article_embeddings_with_padding, args.output_article_content_embeddings)

    #Ps: To experiment with these doc2vec embeddings, it is necessary to deserialize "acr_articles_metadata_embeddings.pickle", substitute the content_article_embedding and serialize for further usage by NAR module
    #This is made by acr_module/notebooks/ACR_Results_Visualization_Gcom_doc2vec.ipynb

if __name__ == '__main__':
    main()


'''
DATA_DIR=/media/data/projects/personal/doutorado/adressa_news/data_transformed && \
python3 -m acr.preprocessing.doc2vec_adressa \
    --input_articles_csv_path ${DATA_DIR}/articles_tfrecords_v4_first_12_sent.csv \
    --output_article_content_embeddings ${DATA_DIR}/pickles_v4/article_content_embeddings_v4_doc2vec.pickle

#--input_articles_csv_path ${DATA_DIR}/adressa_articles.csv \
#--output_article_content_embeddings ${DATA_DIR}/pickles/article_content_embeddings_doc2vec.pickle    
'''