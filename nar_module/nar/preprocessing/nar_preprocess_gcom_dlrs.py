import argparse
import pandas as pd
import numpy as np
import glob

import tensorflow as tf

from ..tf_records_management import save_rows_to_tf_record_file, make_sequential_feature
from ..utils import serialize, deserialize

def create_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input_clicks_csv_path_regex', default='',
            help='Input path of the clicks CSV files.')        


    parser.add_argument(
            '--output_sessions_tfrecords_path', default='',
            help='Output path for TFRecords generated with user sessions')

    return parser



def load_sessions_by_hour(clicks_file_path):
    def to_list(series):
        return list(series)

    clicks_hour_df = pd.read_csv(clicks_file_path)
    #Ensuring that sessions are chronologically ordered
    clicks_hour_df.sort_values(['session_start', 'click_timestamp'], inplace=True)
    sessions_by_hour_df = clicks_hour_df.groupby('session_id').agg({'user_id': min,
                                                                    'session_start': min,
                                                                    'session_size': min,
                                                                    'click_article_id': to_list,
                                                                    'click_timestamp': to_list,
                                                                    'click_environment': to_list,
                                                                    'click_deviceGroup': to_list
                                                                    }
                                                                   ).reset_index()
    return sessions_by_hour_df


def make_sequence_example(row):
    idx, fields = row

    context_features = {
        'user_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[fields['user_id']])),
        'session_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[fields['session_id']])),
        'session_start': tf.train.Feature(int64_list=tf.train.Int64List(value=[fields['session_start']])),
        'session_size': tf.train.Feature(int64_list=tf.train.Int64List(value=[fields['session_size']]))
    }
    
    context = tf.train.Features(feature=context_features)
    
    sequence_features = {
        'event_timestamp': make_sequential_feature(fields["click_timestamp"]),
        #Categorical features
        'item_clicked': make_sequential_feature(fields["click_article_id"]),
        'environment': make_sequential_feature(fields["click_environment"]),
        'deviceGroup': make_sequential_feature(fields["click_deviceGroup"]),
    }    

    sequence_feature_lists = tf.train.FeatureLists(feature_list=sequence_features)
    
    return tf.train.SequenceExample(feature_lists=sequence_feature_lists,
                                    context=context
                                   ) 



def main():
    parser = create_args_parser()
    args = parser.parse_args()

    print('Loading sessions by hour')
    clicks_hour_files = sorted(glob.glob(args.input_clicks_csv_path_regex))

    print('Exporting sessions by hour to TFRecords: {}'.format(args.output_sessions_tfrecords_path))   
    #Exporting a TFRecord for each CSV clicks file (one by hour)
    for hour_index, clicks_hour_file_path in enumerate(clicks_hour_files):
        sessions_by_hour_df = load_sessions_by_hour(clicks_hour_file_path)
        save_rows_to_tf_record_file(sessions_by_hour_df.iterrows(), 
                            make_sequence_example,
                            export_filename=args.output_sessions_tfrecords_path.replace('*', '{0:03d}').format(hour_index))

        if hour_index % 10 == 0:
            print('Exported {} TFRecord files'.format(hour_index))
    
if __name__ == '__main__':
    main()