import tensorflow as tf
import multiprocessing

from .utils import merge_two_dicts, get_tf_dtype


#CONTEXT_FEATURES = ['article_id', 'publisher_id', 'category_id', 'created_at_ts', 'text_length']

def parse_sequence_example(example, features_config, truncate_sequence_length=300):
    # Define how to parse the example
    '''
    context_features = {
        "article_id": tf.FixedLenFeature([], dtype=tf.int64),
        "publisher_id": tf.FixedLenFeature([], dtype=tf.int64),
        "category_id": tf.FixedLenFeature([], dtype=tf.int64),
        "created_at_ts": tf.FixedLenFeature([], dtype=tf.int64),
        "text_length": tf.FixedLenFeature([], dtype=tf.int64),
    }
    '''

    single_features = {feature_name: tf.FixedLenFeature([], dtype=get_tf_dtype(features_config['single_features'][feature_name]['dtype'])) \
                        for feature_name in features_config['single_features']}

    sequence_features = {feature_name: tf.FixedLenSequenceFeature(shape=[], dtype=get_tf_dtype(features_config['sequence_features'][feature_name]['dtype'])) \
                        for feature_name in features_config['sequence_features']}


    single_parsed, sequence_parsed = tf.parse_single_sequence_example(
        example, 
        sequence_features=sequence_features,
        context_features=single_features,
        example_name="example"
    )

    #Truncating max text size
    sequence_parsed['text'] = sequence_parsed['text'][:truncate_sequence_length] 
    
    merged_features = merge_two_dicts(single_parsed, sequence_parsed)

    #In order the pad the dataset, I had to use this hack to expand scalars to vectors.
    expand_features(merged_features, feature_to_expand=features_config['single_features'].keys())

    return merged_features


def expand_features(features, feature_to_expand):
    '''
    Hack. Because padded_batch doesn't play nice with scalres, so we expand the scalar to a vector of length 1
    '''
    for feature_key in feature_to_expand:
        features[feature_key] = tf.expand_dims(tf.convert_to_tensor(features[feature_key]), -1)

def deflate_features(features, feature_to_deflate):
    '''
        Undo Hack. We undo the expansion we did in expand
    '''    
    for feature_key in feature_to_deflate:
        features[feature_key] = tf.squeeze(features[feature_key], axis=[-1])

def get_label_features(features, features_config):
    return {feature_name: features[feature_name] \
            for feature_name in features \
            if feature_name in features_config['label_features']}

def deflate_and_split_features_label(features, features_config):
    #Undo that hack required for padding 
    deflate_features(features, features_config['single_features'])
    labels = get_label_features(features, features_config)

    #Returning features and label separatelly
    return(features, labels)


def make_dataset(path, features_config, batch_size=128, num_map_threads=None, truncate_sequence_length=300):
    '''
    Makes  a Tensorflow dataset that is shuffled, batched and parsed 
    You can chain all the lines here, I split them into seperate calls so I could comment easily
    :param path: The path to a tf record file
    :param path: The size of our batch
    :return: a Dataset that shuffles and is padded
    '''

    if not num_map_threads:
        num_map_threads = multiprocessing.cpu_count()
        tf.logging.info('Using {} threads for parallel map'.format(num_map_threads))


    # Read a tf record file. This makes a dataset of raw TFRecords
    dataset = tf.data.TFRecordDataset(path, compression_type='GZIP')
    # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
    dataset = dataset.map(lambda x: parse_sequence_example(x, features_config,
                                            truncate_sequence_length=truncate_sequence_length), 
                         num_parallel_calls=num_map_threads)

    features_shapes_single = {key: 1 for key in features_config['single_features']}
    features_shapes_sequence = {key: tf.TensorShape([None]) for key in features_config['sequence_features']}
    features_shapes = merge_two_dicts(features_shapes_single, features_shapes_sequence)

    #Batch the dataset so that we get batch_size examples in each batch.
    #Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly
    dataset = dataset.padded_batch(batch_size, padded_shapes=features_shapes)

    '''
    #Batch the dataset so that we get batch_size examples in each batch.
    #Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly
    dataset = dataset.padded_batch(batch_size, padded_shapes={
        "article_id": 1, #Context doesn't need any padding, its always length one
        "publisher_id": 1, 
        "category_id": 1,
        "created_at_ts": 1,
        "text_length": 1,    
        "text": tf.TensorShape([None]), # but the seqeunce is variable length, we pass that information to TF        
    })
    '''

    #Finally, we need to undo that hack from the expand function
    #dataset= dataset.map(deflate)
    #Splitting features and label
    dataset = dataset.map(lambda features: deflate_and_split_features_label(features, features_config), 
                          num_parallel_calls=num_map_threads)
    #Pre-fetches rows ahead
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset


def prepare_dataset_iterator_with_initializer(files, features_config, batch_size=128, truncate_tokens_length=300):
    # Make a dataset 
    ds = make_dataset(files, features_config, batch_size=batch_size,
                      truncate_sequence_length=truncate_tokens_length)
    
    # Define an abstract iterator that has the shape and type of our datasets
    iterator = tf.data.Iterator.from_structure(ds.output_types,
                                               ds.output_shapes)

    # This is an op that gets the next element from the iterator
    next_element = iterator.get_next()
    
    # These ops let us switch and reinitialize every time we finish an epoch    
    iterator_init_op = iterator.make_initializer(ds)

    return next_element, iterator_init_op



def prepare_dataset(files, features_config, batch_size=128, epochs=1, 
                    shuffle_dataset=True, shuffle_buffer_size=3000,
                    truncate_tokens_length=300):
    #Making sure that data preprocessing steps (I/O bound) are not performed on GPU
    with tf.device('/cpu:0'):
        # Make a dataset 
        ds = make_dataset(files, features_config, batch_size=batch_size,
                            truncate_sequence_length=truncate_tokens_length)    
        
        ds = ds.repeat(epochs)
        if shuffle_dataset:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Define an abstract iterator that has the shape and type of our datasets
        iterator = ds.make_one_shot_iterator()

        # This is an op that gets the next element from the iterator
        next_element = iterator.get_next()

        return next_element