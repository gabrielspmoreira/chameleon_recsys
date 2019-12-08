from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle
from time import time
import hashlib
import urllib.parse
import re
import unicodedata
import pytz
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ua_parser import user_agent_parser

def serialize(filename, obj):
    #with open(filename, 'wb') as handle:
    with tf.gfile.Open(filename, 'wb') as handle:
        pickle.dump(obj, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
        
def deserialize(filename):
    #with open(filename, 'rb') as handle:
    with tf.gfile.Open(filename, 'rb') as handle:
        return pickle.load(handle)

def merge_two_dicts(x, y):
    #Python 2 to 3.4
    #z = x.copy()   # start with x's keys and values
    #z.update(y)    # modifies z with y's keys and values & returns None
    #return z
    #Python 3.5 or greater
    return {**x, **y}

def log_elapsed_time(start_time, text=''):
    took = (time() - start_time) / 60.0
    tf.logging.info('==== {} elapsed {:.1f} minutes'.format(text, took))


def resolve_files(regex):
    """Return list of files given a regex"""
    list_op = tf.train.match_filenames_once(regex)
    init_ops = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_ops)
        files = sess.run(list_op)

    return list(sorted(files))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]    


def get_tf_dtype(dtype):
    if dtype == 'int':
        tf_dtype = tf.int64
    elif dtype == 'float':
        tf_dtype = tf.float32
    elif dtype == 'string' or dtype == 'bytes':
        tf_dtype = tf.string
    else:
        raise Exception('Invalid dtype "{}"'.format(dtype))
    return tf_dtype        


def get_pad_token():
    PAD_TOKEN = '<PAD>'
    return PAD_TOKEN
 
def get_unfrequent_token():
    UNFREQ_TOKEN = '<UNF>'
    return UNFREQ_TOKEN

def get_categ_encoder_from_values(values, include_pad_token=True, include_unfrequent_token=False):
    encoder_values = []
    if include_pad_token:
        encoder_values.append(get_pad_token())
    if include_unfrequent_token:
        encoder_values.append(get_unfrequent_token())
    encoder_values.extend(values)
    encoder_ids = list(range(len(encoder_values)))
    encoder_dict = dict(zip(encoder_values, encoder_ids))
    return encoder_dict

def encode_categ_feature(value, encoder_dict):
    if value in encoder_dict:
        return encoder_dict[value]
    else:
        return encoder_dict[get_unfrequent_token()]
        

def max_n_sparse_indexes(row_data, row_indices, topn):
    i = row_data.argsort()[-topn:][::-1]
    top_values = row_data[i]
    top_indices = row_indices[i]  
    return top_indices#, top_values

#Returns a tensor with 2 dimensions with all paired permutations (of size 2) of tensor x
def paired_permutations(x):
    #Ensuring the vector is flatten
    #x = tf.reshape(x, [-1])    
    size = tf.shape(x)[0]

    counter = tf.constant(0)
    m0 = tf.zeros(shape=[0, 2], dtype=x.dtype)
    cond = lambda i,m: i < size*size
    body = lambda i,m: [i+1, tf.concat([m, tf.expand_dims(tf.stack([x[tf.to_int32(tf.div(i,size))], 
                                                                    x[tf.mod(i,size)]])
                                                          , axis=0)
                                       ], axis=0, name="concat_rows")
                       ]
    _, combined_values = tf.while_loop(
        cond, body, 
        loop_vars=[counter, m0],
        shape_invariants=[counter.get_shape(), tf.TensorShape([None,None])])
    return combined_values


def get_days_diff(newer_timestamp, older_timestamp):
    sec_diff = newer_timestamp - older_timestamp
    days_diff = sec_diff / 60 / 60 / 24
    return days_diff

def get_time_decay_factor(newer_timestamp, older_timestamp, alpha=0.5):
    days_diff = get_days_diff(newer_timestamp, older_timestamp)
    denominator = math.pow(1+alpha, days_diff)
    if denominator != 0:
        return 1.0 / denominator
    else:
        return 0.0

def append_lines_to_text_file(filename, lines):
    with open(filename, "a") as myfile:
        myfile.writelines([line+"\n" for line in lines])

def hash_str_to_int(encoded_bytes_text, digits):
    return int(str(int(hashlib.md5(encoded_bytes_text).hexdigest()[:8], 16))[:digits])        


def get_os_list():
    return ['iOS',
           'Android',
           'Windows Phone',
           'Windows Mobile',
           'Windows',
           'Mac OS X',
           'Mac OS',
           'Samsung',
           'FireHbbTV',
           'ATV OS X',
           'tvOS',
           'Chrome OS',
           'Debian',
           'Symbian OS',
           'BlackBerry OS',
           'Firefox OS',
           'Android',
           'Brew MP',
           'Chromecast',
           'webOS',
           'Gentoo',
           'Solaris']

def extract_os_from_user_agent(user_agent, default_os='Other'):
    parsed_os = user_agent_parser.ParseOS(user_agent)
    os_family = parsed_os['family']
    if 'Symbian' in os_family:
        os_family = 'Symbian OS'
    elif 'BlackBerry' in os_family:
        os_family = 'BlackBerry OS'

    if os_family is None or os_family not in get_os_list():
        os_family = default_os

    return os_family


domain_pattern = re.compile(r"^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/\n]+)")
def extract_domain_from_url(url):
    s = domain_pattern.search(url)    
    if s is None:
        return None
    else:
        domain = s.group(0)
        return domain


def urlencode(str):
  return urllib.parse.quote(str)


def urldecode(str):
  return urllib.parse.unquote(str)

def extract_local_hour_weekday(timestamp_in_utc, local_tz):
    dt = pytz.utc.localize(datetime.datetime.utcfromtimestamp(timestamp_in_utc)).astimezone(pytz.timezone(local_tz))
    return dt.hour + (dt.minute/60.0), dt.weekday() #First day is Monday


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def gini_index(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))                  



def min_max_scale(vector, min_max_range=(-1.0,1.0)):
    scaler = MinMaxScaler(feature_range=min_max_range)
    norm_vector = scaler.fit_transform(vector)
    return norm_vector



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')    