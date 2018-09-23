from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import multiprocessing

from .utils import merge_two_dicts, get_tf_dtype

def expand_single_features(x, features_to_expand):
        '''
        Hack. Because padded_batch doesn't play nice with scalres, so we expand the scalar to a vector of length 1
        '''
        for feature_key in features_to_expand:
            x[feature_key] = tf.expand_dims(tf.convert_to_tensor(x[feature_key]), 0)

        return x


def deflate_single_features(x, expanded_features):
    '''
        Undo Hack. We undo the expansion we did in expand and make sure that vector has rank 2 (adds one dimension if this batch size == 1)
    '''    
    for feature_key in expanded_features:
        x[feature_key] = expand_to_vector_if_scalar(tf.squeeze(x[feature_key]))

    return x

def expand_to_vector_if_scalar(tensor):
    return tf.cond(tf.logical_and(tf.equal(tf.size(tensor), tf.constant(1)),
                                  tf.equal(tf.rank(tensor), tf.constant(0))),
                   lambda: tf.expand_dims(tensor, 0),
                   lambda: tensor)

def parse_sequence_example(example, features_config, truncate_sequence_length=20):

    # Define how to parse the example
    context_features = {}
    features_config_single = features_config['single_features']
    for feature_name in features_config_single:        
        context_features[feature_name] = tf.FixedLenFeature([], 
                      dtype=get_tf_dtype(features_config_single[feature_name]['dtype']))

    
    sequence_features = {}
    features_config_sequence = features_config['sequence_features']
    for feature_name in features_config_sequence:        
        sequence_features[feature_name] = tf.FixedLenSequenceFeature(shape=[], 
                      dtype=get_tf_dtype(features_config_sequence[feature_name]['dtype']))

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        example, 
        sequence_features=sequence_features,
        context_features=context_features,
        example_name="example"
    )


    #Truncate long sessions to a limit
    context_parsed['session_size'] = tf.minimum(context_parsed['session_size'], 
                                                truncate_sequence_length)
    for feature_name in sequence_parsed:
        sequence_parsed[feature_name] = sequence_parsed[feature_name][:truncate_sequence_length] 
    
    
    #Ignoring first click from labels
    sequence_parsed['label_next_item'] = sequence_parsed['item_clicked'][1:]    
    #Making it easy to retrieve the last label
    sequence_parsed['label_last_item'] = sequence_parsed['item_clicked'][-1:]
    
    #Ignoring last clicked item from input    
    for feature_key in sequence_features:
        if feature_key not in ['label_next_item', 'label_last_item']:
            sequence_parsed[feature_key] = sequence_parsed[feature_key][:-1]
    
    merged_features = merge_two_dicts(context_parsed, sequence_parsed)

    #In order the pad the dataset, I had to use this hack to expand scalars to vectors.
    merged_expanded_features = expand_single_features(merged_features,
                                      features_to_expand=list(features_config['single_features'].keys()))

    return merged_expanded_features

def deflate_and_split_features_label(x, expanded_features):    
    #Undo that hack required for padding 
    x = deflate_single_features(x, expanded_features)

    labels = {
        'label_next_item': x['label_next_item'],
        'label_last_item': x['label_last_item']
    }
    
    del x['label_next_item']
    del x['label_last_item']
    
    #Returning features and label separatelly
    return(x, labels)


def make_dataset(path, features_config, batch_size=128, num_map_threads=None,
                 truncate_sequence_length=20):

    def get_features_shapes(features_config):
        features_shapes = {}

        for feature_name in features_config['single_features']:        
            features_shapes[feature_name] = 1

        for feature_name in features_config['sequence_features']:        
            features_shapes[feature_name] = tf.TensorShape([None])

        features_shapes['label_next_item'] = tf.TensorShape([None])
        features_shapes['label_last_item'] = tf.TensorShape([None])

        return features_shapes


    if not num_map_threads:
        num_map_threads = multiprocessing.cpu_count()
        tf.logging.info('Using {} threads for parallel map'.format(num_map_threads))


    # Read a tf record file. This makes a dataset of raw TFRecords
    dataset = tf.data.TFRecordDataset(path, compression_type='GZIP')
    # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
    dataset =  dataset.map(lambda x: parse_sequence_example(x, features_config,
                                                            truncate_sequence_length=truncate_sequence_length), 
                                            num_parallel_calls=num_map_threads)


    
    #Batch the dataset so that we get batch_size examples in each batch.
    #Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly    
    features_shapes = get_features_shapes(features_config)
    dataset = dataset.padded_batch(batch_size, padded_shapes=features_shapes)

    #Splitting features and label
    expanded_features=list(features_config['single_features'].keys())
    dataset = dataset.map(lambda x: deflate_and_split_features_label(x, expanded_features), 
                          num_parallel_calls=num_map_threads)
    #Pre-fetches one batch ahead
    dataset = dataset.prefetch(1)
    return dataset


def prepare_dataset_iterator_with_init(files, features_config, batch_size=128, 
                                        truncate_session_length=20):
    # Make a dataset 
    ds = make_dataset(files, features_config, batch_size=batch_size,
                        truncate_sequence_length=truncate_session_length)
    
    # Define an abstract iterator that has the shape and type of our datasets
    iterator = tf.data.Iterator.from_structure(ds.output_types,
                                               ds.output_shapes)

    # This is an op that gets the next element from the iterator
    next_element = iterator.get_next()
    
    # These ops let us switch and reinitialize every time we finish an epoch    
    iterator_init_op = iterator.make_initializer(ds)

    return next_element, iterator_init_op



def prepare_dataset_iterator(files, features_config, batch_size=128,
                            truncate_session_length=20):
    with tf.device('/cpu:0'):
        # Make a dataset 
        ds = make_dataset(files, features_config, batch_size=batch_size,
                            truncate_sequence_length=truncate_session_length)    
        
        # Define an abstract iterator that has the shape and type of our datasets
        iterator = ds.make_one_shot_iterator()

        # This is an op that gets the next element from the iterator
        next_element = iterator.get_next()

        return next_element