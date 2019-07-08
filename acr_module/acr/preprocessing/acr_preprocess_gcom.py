import argparse
import pandas as pd
import re
import nltk
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from ..tf_records_management import export_dataframe_to_tf_records, make_sequential_feature
from ..utils import serialize
from .tokenization import tokenize_articles, nan_to_str, convert_tokens_to_int, get_words_freq
from .word_embeddings import load_word_embeddings, process_word_embedding_for_corpus_vocab, save_word_vocab_embeddings


def create_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input_articles_csv_path', default='',
            help='Input path of the news CSV file.')

    parser.add_argument(
            '--input_word_embeddings_path', default='',
            help='Input path of the word2vec embeddings model (word2vec).')    

    parser.add_argument(
            '--output_tf_records_path', default='',
            help='Output path for generated TFRecords with news content.')

    parser.add_argument(
            '--output_word_vocab_embeddings_path', default='',
            help='Output path for a pickle with words vocabulary and corresponding word embeddings.')

    parser.add_argument(
            '--output_label_encoders', default='',
            help='Output path for a pickle with label encoders (article_id, category_id, publisher_id).')

    parser.add_argument(
        '--articles_by_tfrecord', type=int, default=1000,
        help='Number of articles to be exported in each TFRecords file')

    parser.add_argument(
        '--vocab_most_freq_words', type=int, default=100000,
        help='Most frequent words to keep in vocab')

    return parser


#############################################################################################
#Based on text cleaner used to generate Brazilian Portuguese word embeddings:
#https://github.com/nathanshartmann/portuguese_word_embeddings/blob/master/preprocessing.py

# Punctuation list
punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')

re_remove_brackets = re.compile(r'\{.*\}')
re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
re_transform_numbers = re.compile(r'\d', re.UNICODE)
re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
# Different quotes are used.
re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
re_tree_dots = re.compile(u'…', re.UNICODE)
# Differents punctuation patterns are used.
re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                       (punctuations, punctuations), re.UNICODE)
re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                         (punctuations, punctuations), re.UNICODE)
re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
re_changehyphen = re.compile(u'–')
re_doublequotes_1 = re.compile(r'(\"\")')
re_doublequotes_2 = re.compile(r'(\'\')')
re_trim = re.compile(r' +', re.UNICODE)


def clean_str(string):
    string = string.replace('\n', ' ')
    """Apply all regex above to a given string."""
    string = string.lower()
    string = re_tree_dots.sub('...', string)
    string = re.sub('\.\.\.', '', string)
    string = re_remove_brackets.sub('', string)
    string = re_changehyphen.sub('-', string)
    string = re_remove_html.sub(' ', string)
    string = re_transform_numbers.sub('0', string)
    string = re_transform_url.sub('URL', string)
    string = re_transform_emails.sub('EMAIL', string)
    string = re_quotes_1.sub(r'\1"', string)
    string = re_quotes_2.sub(r'"\1', string)
    string = re_quotes_3.sub('"', string)
    string = re.sub('"', '', string)
    string = re_dots.sub('.', string)
    string = re_punctuation.sub(r'\1', string)
    string = re_hiphen.sub(' - ', string)
    string = re_punkts.sub(r'\1 \2 \3', string)
    string = re_punkts_b.sub(r'\1 \2 \3', string)
    string = re_punkts_c.sub(r'\1 \2', string)
    string = re_doublequotes_1.sub('\"', string)
    string = re_doublequotes_2.sub('\'', string)
    string = re_trim.sub(' ', string)
        
    return string.strip()


sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
def clean_and_filter_first_sentences(string, first_sentences=8):
    # Tokenize sentences and remove short and malformed sentences.
    sentences = []
    for sent in sent_tokenizer.tokenize(string):
        if sent.count(' ') >= 3 and sent[-1] in ['.', '!', '?', ';']:
            sentences.append(clean_str(sent))
            if len(sentences) == first_sentences:
                break
    return ' '.join(sentences)

#############################################################################################

def load_input_csv(path):
    news_df = pd.read_csv(path, encoding = 'utf-8')

    #Concatenating all available text
    news_df['full_text'] = (news_df['title'].apply(nan_to_str) + ". " + \
                            news_df['caption'].apply(nan_to_str) + ". " + \
                            news_df['body'].apply(nan_to_str)
                       ).apply(clean_and_filter_first_sentences)

    return news_df

def process_cat_features(dataframe):
    article_id_encoder = LabelEncoder()
    dataframe['id_encoded'] = article_id_encoder.fit_transform(dataframe['id'])

    category_id_encoder = LabelEncoder()
    dataframe['categoryid_encoded'] = category_id_encoder.fit_transform(dataframe['categoryid'])

    domainid_encoder = LabelEncoder()
    dataframe['domainid_encoded'] = domainid_encoder.fit_transform(dataframe['domainid'])

    return article_id_encoder, category_id_encoder, domainid_encoder

def save_article_cat_encoders(output_path, article_id_encoder, category_id_encoder, domainid_encoder):
    to_serialize = {'article_id': article_id_encoder, 
                    'category_id': category_id_encoder, 
                    'publisher_id': domainid_encoder}
    serialize(output_path, to_serialize)


def make_sequence_example(row):
    context_features = {
        'article_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['id_encoded']])),
        'publisher_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['domainid_encoded']])),
        'category_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['categoryid_encoded']])),
        'created_at_ts': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['created_at_ts']])),
        'text_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['text_length']]))
    }
    
    context = tf.train.Features(feature=context_features)
    
    sequence_features = {
        'text': make_sequential_feature(row["text_int"], vtype=int)        
    }    

    sequence_feature_lists = tf.train.FeatureLists(feature_list=sequence_features)
    
    return tf.train.SequenceExample(feature_lists=sequence_feature_lists,
                                    context=context
                                   )    

def main():
    parser = create_args_parser()
    args = parser.parse_args()

    print('Loading news article CSV: {}'.format(args.input_articles_csv_path))
    news_df = load_input_csv(args.input_articles_csv_path)

    print('Encoding categorical features')
    article_id_encoder, category_id_encoder, domainid_encoder = process_cat_features(news_df)
    print('Exporting LabelEncoders of categorical features: {}'.format(args.output_label_encoders))
    save_article_cat_encoders(args.output_label_encoders, 
                              article_id_encoder, 
                              category_id_encoder, 
                              domainid_encoder)
    
    print('Tokenizing articles...')
    tokenized_articles = tokenize_articles(news_df['full_text'])

    print('Computing word frequencies...')
    words_freq = get_words_freq(tokenized_articles)
    print('Corpus vocabulary size: {}'.format(len(words_freq)))

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

    data_to_export_df = news_df[['id_encoded', 
                                 'domainid_encoded', 
                                 'categoryid_encoded', 
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