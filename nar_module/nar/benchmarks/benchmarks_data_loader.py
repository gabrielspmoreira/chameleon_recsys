from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
from enum import Enum

from ..datasets import make_dataset

def load_eval_negative_samples(eval_sessions_negative_samples_json_path):
    eval_sessions_neg_samples_df = pd.read_json(eval_sessions_negative_samples_json_path, lines=True,
                                                 dtype={'session_id': np.int64})
    eval_sessions_neg_samples = dict(eval_sessions_neg_samples_df[['session_id', 'negative_items']].values)
    return eval_sessions_neg_samples




G1_DATASET = "gcom"
ADRESSA_DATASET = "adressa"

class DataLoader:

	def __init__(self, dataset):
		features_config = self.get_session_features_config(dataset)
		self.init_dataset_iterator_local(features_config, batch_size=1)


	def get_session_features_config(self, dataset):
		user_id_type = 'int' if dataset == G1_DATASET else 'bytes'		
		session_features_config = {
	        'single_features': {
	            ##Control features
	            'user_id': {'type': 'categorical', 'dtype': user_id_type},  
	            'session_id': {'type': 'categorical', 'dtype': 'int'},
	            #'session_id': {'type': 'categorical', 'dtype': 'string'},            
	            'session_start': {'type': 'categorical', 'dtype': 'int'},
	            'session_size': {'type': 'categorical', 'dtype': 'int'},
	        },
	        'sequence_features': {
	            #Required sequence features
	            'event_timestamp': {'type': 'categorical', 'dtype': 'int'},
	            'item_clicked': {'type': 'categorical', 'dtype': 'int'}, #, 'cardinality': 364047},           
	        }
	    }

		return session_features_config

	def init_dataset_iterator_local(self, features_config, batch_size=128, 
                                        truncate_session_length=20):
	    with tf.device('/cpu:0'):
	        self.files_placeholder = tf.placeholder(tf.string)

	        # Make a dataset 
	        ds = make_dataset(self.files_placeholder, features_config, batch_size=batch_size,
	                            truncate_sequence_length=truncate_session_length)

	        
	        # Define an abstract iterator that has the shape and type of our datasets
	        iterator = tf.data.Iterator.from_structure(ds.output_types,
	                                                   ds.output_shapes)

	        # This is an op that gets the next element from the iterator
	        self.next_element_op = iterator.get_next()
	        
	        # These ops let us switch and reinitialize every time we finish an epoch    
	        self.iterator_init_op = iterator.make_initializer(ds)


	def load_dataframe(self, data_filenames):
	    data = []

	    session_cnt = 0
	    repeated = 0
	    with tf.Session() as sess:  
	        sess.run(self.iterator_init_op, feed_dict={self.files_placeholder: data_filenames})
	        while True:  
	            try:
	                #One session by batch
	                batch_inputs, batch_labels = sess.run(self.next_element_op)

	                item_ids_session = set()

	                session_id = batch_inputs['session_id'][0]
	                for item, ts in zip(batch_inputs['item_clicked'][0], batch_inputs['event_timestamp'][0]):
	                    if item in item_ids_session:
	                        repeated += 1
	                    item_ids_session.add(item)
	                    data.append((session_id, item, ts))

	                #Adding last item (label)
	                last_item = batch_labels['label_last_item'][0][0]
	                if last_item in item_ids_session:
	                    repeated += 1
	                data.append((session_id, last_item, ts))

	                session_cnt += 1
	                #if cnt % 100 == 0:
	                #    print("Sessions processed: {} - Clicks: {}".format(session_cnt, len(data)))
	            except tf.errors.OutOfRangeError as e:
	                break

	    if len(data) > 0:
	        print("Sessions read: {} - Clicks: {} - Repeated Clicks: {}".format(session_cnt, len(data), repeated))
	    else:
	        print('WARNING: NO DATA FOUND!')
	    data_df = pd.DataFrame(data, columns=['SessionId', 'ItemId', 'Time'])
	    return data_df