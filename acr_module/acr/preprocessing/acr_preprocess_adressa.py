import argparse
import pandas as pd
import os
from collections import defaultdict, Counter
import json
import numpy as np
import nltk
from dateutil.parser import parse
from joblib import Parallel, delayed
from sklearn.utils import class_weight

import tensorflow as tf

from ..tf_records_management import export_dataframe_to_tf_records, make_sequential_feature
from ..utils import serialize, chunks, get_categ_encoder_from_values, encode_categ_feature
from .tokenization import tokenize_articles, nan_to_str, convert_tokens_to_int, get_words_freq
from .word_embeddings import load_word_embeddings, process_word_embedding_for_corpus_vocab, save_word_vocab_embeddings

def create_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input_articles_folder_path', default='',
            help='Input Adressa contentdata folder path.')

    parser.add_argument(
            '--input_word_embeddings_path', default='',
            help='Input path of the word2vec embeddings model (word2vec).') 

    parser.add_argument(
            '--max_words_length', type=int, default=1000,
            help='Maximum tokens length of text.')   

    parser.add_argument(
            '--output_tf_records_path', default='',
            help='Output path for generated TFRecords with news content.')

    parser.add_argument(
            '--output_word_vocab_embeddings_path', default='',
            help='Output path for a pickle with words vocabulary and corresponding word embeddings.')

    parser.add_argument(
            '--output_label_encoders', default='',
            help='Output path for a pickle with label encoders for categorical features.')

    parser.add_argument(
            '--output_articles_csv_path', default='',
            help='Output path for a CSV file with articles contents.')   

    parser.add_argument(
        '--articles_by_tfrecord', type=int, default=1000,
        help='Number of articles to be exported in each TFRecords file')

    parser.add_argument(
        '--vocab_most_freq_words', type=int, default=100000,
        help='Most frequent words to keep in vocab')

    return parser

parser = create_args_parser()
args = parser.parse_args()


CATEGORIES_TO_IGNORE = ['bolig', 'abonnement']
SITES_TO_IGNORE = ['kundeservice.adressa.no']


def unique_list_if_str(value):
    if type(value) == list:
        return value
    else:
        return [value]  

def parse_content_general(line):
    content_raw = json.loads(line)
    
    new_content = {}
    for key in content_raw:
        if key == 'fields':
            for field in content_raw['fields']:
                value = field['value']
                if field['field'] == 'body':
                    value = ' '.join(value)
                new_content[field['field']] = value
        else:
            new_content[key] = content_raw[key]
    
        
    return new_content

    
def parse_content(line):
    content_json = parse_content_general(line)  
    content_raw = defaultdict(str, content_json)

    publishtime = content_raw['publishtime'] if content_raw['publishtime'] != '' else content_raw['createtime'] 
    #Converting to unix timestamp in miliseconds
    publishtime_ts = int(parse(publishtime).timestamp()) * 1000
    
    author_1st = content_raw['author'][0] if type(content_raw['author']) == list else content_raw['author']
    
    if type(content_raw['heading']) == list:
        heading = set(content_raw['heading']) #Set to remove repeated phrases
    else:
        heading = [content_raw['heading']]
    
    textual_highlights = "{} | {} | {} | {}".format(content_raw['title'], 
                                                            content_raw['teaser'], 
                                                            '. '.join(heading),
                                                            content_raw['body']) \
                        .replace(u'\xad','').replace('"', '')
    
    new_content = {'id': content_raw['id'],
                   'url': content_raw['url'],
                   'site': unique_list_if_str(content_raw['og-site-name'])[0],
                   'adressa-access': content_raw['adressa-access'], #(free, subscriber)
                   'author_1st':  author_1st if author_1st != '' else '', #3777 unique                  
                   'publishtime': publishtime,
                   'created_at_ts': publishtime_ts,
                   'text_highlights': textual_highlights, 
                   #Extracted using NLP techniques (by Adressa)
                   'concepts': ','.join(unique_list_if_str(content_raw['kw-concept'])), #98895 unique
                   'entities': ','.join(unique_list_if_str(content_raw['kw-entity'])), #150214 unique
                   'locations': ','.join(unique_list_if_str(content_raw['kw-location'])), #5533 unique
                   'persons': ','.join(unique_list_if_str(content_raw['kw-person'])), #53535 unique
                   #Categories and keywords tagged by the journalists of Adresseavisen and may be of variable quality (label)
                   'category0': content_raw['category0'], #39 unique
                   'category1': content_raw['category1'] if 'category1' in content_raw else '', #126 unique
                   'category2': content_raw['category2'] if 'category2' in content_raw else '', #75 unique
                   'keywords': content_raw['keywords'], #6489 unique
                  }

        
    return new_content

def parse_content_file(file_path, parser_fn):
    with open(file_path, 'r', encoding='utf-8') as fi:
        try:
            for line in fi:
                if line.strip() == 'null':
                    return None
                content = parser_fn(line)
                #Returns only the first json from content file, as others are identical, only "score" is different
                return content
        except Exception as e:
            print('Error processing file "{}": {}'.format(file_path, e))
            raise

def load_contents_from_files_list(root_path, files_list):
    total_contents = 0
    invalid_contents_count = 0
    articles = []

    for idx, filename in enumerate(files_list):                
        file_content = parse_content_file(os.path.join(root_path, filename), parse_content)
        if file_content == None:
            #print('File content is null: {}'.format(filename))
            invalid_contents_count += 1
        else:
            articles.append(file_content)
        total_contents += 1
    
    print("Processed content files: {} - Empty files: {} - Valid articles: {}".format(total_contents, invalid_contents_count, len(articles)))
    return articles

FILES_BY_CHUNK = 5000
JOBS_TO_LOAD_FILES = 4
def load_contents_from_folder(path):
    articles_files_chunks = chunks(sorted(os.listdir(path)), FILES_BY_CHUNK)
    #articles_files_chunks = [list(articles_files_chunks)[0]]

    #Starting 4 processes to speed up files parsing
    articles_chunks = Parallel(n_jobs=JOBS_TO_LOAD_FILES)(delayed(load_contents_from_files_list)(path, files_list) for files_list in articles_files_chunks)

    #Merging articles in a data frame
    news_df = pd.DataFrame([articles for chunk in articles_chunks for articles in chunk])

    #Filtering out news with invalid categories or from specific sites
    news_df = news_df[(~news_df['category0'].isin(CATEGORIES_TO_IGNORE)) & \
                            (~news_df['site'].astype(str).isin(SITES_TO_IGNORE))]

    news_df.drop_duplicates(subset='id', keep='first', inplace=True)
    return news_df

def flatten_list_series(series_of_lists):
    return pd.DataFrame(series_of_lists.apply(pd.Series).stack().reset_index(name='item'))['item']

def get_freq_values(series, min_freq=100):
    flatten_values_counts = series.groupby(series).size()
    return flatten_values_counts[flatten_values_counts >= min_freq].sort_values(ascending=False).reset_index(name='count')

def get_freq_values_series_of_lists(series_of_lists, min_freq=100):
    flatten_values = flatten_list_series(series_of_lists)
    flatten_values_counts = get_freq_values(flatten_values)
    return flatten_values_counts

PAD_TOKEN = '<PAD>'
def include_pad_token(unique_values):    
        return np.hstack([[PAD_TOKEN], unique_values])

UNFREQ_TOKEN = "<UNF>" #Unfrequent value    
def include_unfrequent_token(unique_values):    
        return np.hstack([[UNFREQ_TOKEN], unique_values])   


def get_encoder_from_freq_values(series, min_freq=100):
    freq_values_counts_df = get_freq_values(series, min_freq=min_freq)
    encoder = get_categ_encoder_from_values(freq_values_counts_df[freq_values_counts_df.columns[0]].unique(), include_unfrequent_token=True)    
    return encoder

def transform_categorical_column(series, encoder):
    return series.apply(lambda x: encode_categ_feature(x, encoder)) 

def get_encoder_from_freq_values_in_list_column(series, min_freq=100):
    freq_values_counts_df = get_freq_values_series_of_lists(series, min_freq=min_freq)
    encoder = get_categ_encoder_from_values(freq_values_counts_df[freq_values_counts_df.columns[0]].unique(), include_unfrequent_token=False)
    return encoder

def transform_categorical_list_column(series, encoder):
    return series.apply(lambda l: list([encoder[val] for val in l if val in encoder]))    

def comma_sep_values_to_list(value):
    return list([y.strip() for y in value.split(',') if y.strip() != ''])

def get_sample_weight_inv_freq(class_value, classes_count, numerator=10.0):
    return numerator / classes_count[class_value]

def process_cat_features(news_df):
    article_id_encoder = get_categ_encoder_from_values(news_df['id'])
    print('Articles - unique count {}'.format(len(article_id_encoder)))
    news_df['id_encoded'] = transform_categorical_column(news_df['id'], article_id_encoder)

    category0_encoder = get_categ_encoder_from_values(news_df['category0'].unique())
    print('Category0 - unique count {}'.format(len(category0_encoder)))
    news_df['category0_encoded'] = transform_categorical_column(news_df['category0'], category0_encoder)
    
    category0_class_weights = class_weight.compute_class_weight('balanced', classes=news_df['category0_encoded'].unique(), y=news_df['category0_encoded'])
    print('Category0 weights: {}'.format(category0_class_weights))
    
    category1_encoder = get_categ_encoder_from_values(news_df['category1'].unique())
    print('Category1 - unique count {}'.format(len(category1_encoder)))
    news_df['category1_encoded'] = transform_categorical_column(news_df['category1'], category1_encoder)
    
    category1_class_weights = class_weight.compute_class_weight('balanced', classes=news_df['category1_encoded'].unique(), y=news_df['category1_encoded'])
    print('Category1 weights: {}'.format(category1_class_weights))

    #Including only frequent authors
    author_encoder = get_encoder_from_freq_values(news_df['author_1st'])
    print('Author - freq. unique count {}'.format(len(author_encoder)))
    news_df['author_encoded'] = transform_categorical_column(news_df['author_1st'], author_encoder)

    #Converting values separated by "," to lists
    news_df['keywords'] = news_df['keywords'].apply(comma_sep_values_to_list)
    news_df['concepts'] = news_df['concepts'].apply(comma_sep_values_to_list)
    news_df['entities'] = news_df['entities'].apply(comma_sep_values_to_list)
    news_df['locations'] = news_df['locations'].apply(comma_sep_values_to_list)
    news_df['persons'] = news_df['persons'].apply(comma_sep_values_to_list)


    #Processing categorical list columns (Including only frequent categories)
    
    keywords_encoder = get_encoder_from_freq_values_in_list_column(news_df['keywords'])
    news_df['keywords_encoded'] = transform_categorical_list_column(news_df['keywords'], keywords_encoder)
    print('Keywords - freq. unique count {}'.format(len(keywords_encoder)))

    concepts_encoder = get_encoder_from_freq_values_in_list_column(news_df['concepts'])
    news_df['concepts_encoded'] = transform_categorical_list_column(news_df['concepts'], concepts_encoder)
    print('Concepts - freq. unique count {}'.format(len(concepts_encoder)))

    entities_encoder = get_encoder_from_freq_values_in_list_column(news_df['entities'])
    news_df['entities_encoded'] = transform_categorical_list_column(news_df['entities'], entities_encoder)
    print('Entities - freq. unique count {}'.format(len(entities_encoder)))

    locations_encoder = get_encoder_from_freq_values_in_list_column(news_df['locations'])
    news_df['locations_encoded'] = transform_categorical_list_column(news_df['locations'], locations_encoder)
    print('Locations - freq. unique count {}'.format(len(locations_encoder)))

    persons_encoder = get_encoder_from_freq_values_in_list_column(news_df['persons'])
    news_df['persons_encoded'] = transform_categorical_list_column(news_df['persons'], persons_encoder)
    print('Persons - freq. unique count {}'.format(len(persons_encoder)))


    cat_features_encoders = {'article_id': article_id_encoder, 
                    'category0': category0_encoder, 
                    'category1': category1_encoder,
                    'keywords': keywords_encoder,
                    'author': author_encoder,
                    'concepts': concepts_encoder,
                    'entities': entities_encoder,
                    'locations': locations_encoder,
                    'persons': persons_encoder,
                    }

    labels_class_weights = {
                            'category0': category0_class_weights,
                            'category1': category1_class_weights,
                            }

    return cat_features_encoders, labels_class_weights


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
    return words_tokenized[:args.max_words_length]


def save_article_cat_encoders(output_path, cat_features_encoders, labels_class_weights):
    to_serialize = (cat_features_encoders, labels_class_weights)
    serialize(output_path, to_serialize)


def make_sequence_example(row):
    context_features = {
        'article_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['id_encoded']])),
        'category0': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['category0_encoded']])),
        'category1': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['category1_encoded']])),
        'author': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['author_encoded']])),
        'created_at_ts': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['created_at_ts']])),
        'text_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['text_length']])),
        #Only for debug
        'article_id_original': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['id'].encode()])),
        'url': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row['url'].encode()]))
    }
    
    context = tf.train.Features(feature=context_features)
    
    sequence_features = {
        'text': make_sequential_feature(row["text_int"], vtype=int),
        'keywords': make_sequential_feature(row["keywords_encoded"], vtype=int),
        'concepts': make_sequential_feature(row["concepts_encoded"], vtype=int),
        'entities': make_sequential_feature(row["entities_encoded"], vtype=int),
        'locations': make_sequential_feature(row["locations_encoded"], vtype=int),
        'persons': make_sequential_feature(row["persons_encoded"], vtype=int)
    }    

    sequence_feature_lists = tf.train.FeatureLists(feature_list=sequence_features)
    
    return tf.train.SequenceExample(feature_lists=sequence_feature_lists,
                                    context=context
                                   )    

def main():
    

    print('Loading contents from folder: {}'.format(args.input_articles_folder_path))
    news_df = load_contents_from_folder(args.input_articles_folder_path)
    print('Total articles loaded: {}'.format(len(news_df)))

    print('Encoding categorical features')
    cat_features_encoders, labels_class_weights = process_cat_features(news_df)
    
    print('Exporting LabelEncoders of categorical features: {}'.format(args.output_label_encoders))
    save_article_cat_encoders(args.output_label_encoders, cat_features_encoders, labels_class_weights)
    
    print("Saving news articles CSV to {}".format(args.output_articles_csv_path))
    news_df.to_csv(args.output_articles_csv_path, index=False)

    print('Tokenizing articles...')
    tokenized_articles = tokenize_articles(news_df['text_highlights'].values, tokenization_fn=tokenize_norwegian_article)

    print('Computing word frequencies...')
    words_freq = get_words_freq(tokenized_articles)

    print("Loading word2vec model and extracting words of this corpus' vocabulary...")
    w2v_model = load_word_embeddings(args.input_word_embeddings_path, binary=False)
    word_vocab, word_embeddings_matrix = process_word_embedding_for_corpus_vocab(w2v_model, 
                                                                                words_freq,
                                                                                args.vocab_most_freq_words)

    print('Saving word embeddings and vocab.: {}'.format(args.output_word_vocab_embeddings_path))
    save_word_vocab_embeddings(args.output_word_vocab_embeddings_path, 
                               word_vocab, word_embeddings_matrix)

    print('Converting tokens to int numbers (according to the vocab.)...')
    texts_int, texts_lengths = convert_tokens_to_int(tokenized_articles, word_vocab)
    news_df['text_length'] = texts_lengths
    news_df['text_int'] = texts_int

    data_to_export_df = news_df[['id', 'url', #For debug
                                'id_encoded', 
                                'category0_encoded',
                                'category1_encoded',
                                'keywords_encoded',
                                'author_encoded',
                                'concepts_encoded',
                                'entities_encoded',
                                'locations_encoded',
                                'persons_encoded',
                                'created_at_ts',
                                'text_length', 
                                'text_int']]

    print('Exporting tokenized articles to TFRecords: {}'.format(args.output_tf_records_path))                                
    export_dataframe_to_tf_records(data_to_export_df, 
                                   make_sequence_example,
                                   output_path=args.output_tf_records_path, 
                                   examples_by_file=args.articles_by_tfrecord)

if __name__ == '__main__':
    main()