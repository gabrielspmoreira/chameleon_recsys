#WARNING: Clone code
#TODO: Investigate how to share this script between ACR and NAR Python modules 

import sys

import tensorflow as tf
from tensorflow.python.lib.io import tf_record

from .utils import serialize, chunks


def make_sequential_feature(values, vtype=int):
    if vtype == int:
        features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) for value in values]
    elif vtype == float:
        features = [tf.train.Feature(float_list=tf.train.FloatList(value=[value])) for value in values]
    return tf.train.FeatureList(feature=features)
    

def save_rows_to_tf_record_file(df_rows, make_sequence_example_fn, export_filename):
    tf_record_options = tf_record.TFRecordOptions(tf_record.TFRecordCompressionType.GZIP)

    tf_writer = tf_record.TFRecordWriter(export_filename, options=tf_record_options)
    try:
        for index, row in df_rows.iterrows():
            seq_example = make_sequence_example_fn(row)
            tf_writer.write(seq_example.SerializeToString())
    finally:
        tf_writer.close()
        sys.stdout.flush()    

def export_dataframe_to_tf_records(dataframe, make_sequence_example_fn, output_path, 
                                    examples_by_file=1000):        
    export_file_template = output_path.replace('*', '{0:04d}')

    #Exporting rows to TF record by chunks
    for chunk_index, df_chunk in enumerate(chunks(dataframe, examples_by_file)):
        print("Exporting chunk {} (length: {})".format(chunk_index, len(df_chunk)))        
        save_rows_to_tf_record_file(df_chunk, 
                            make_sequence_example_fn,
                            export_filename=export_file_template.format(chunk_index))