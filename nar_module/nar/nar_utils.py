import pandas as pd
import tensorflow as tf
import re
import os

from .utils import deserialize
from .gcs_utils import upload_local_dir_to_gcs, download_from_gcs_dir

def load_acr_module_resources(acr_module_resources_path):
    (acr_label_encoders, articles_metadata_df, content_article_embeddings) = \
              deserialize(acr_module_resources_path)

    tf.logging.info("Read ACR label encoders for: {}".format(acr_label_encoders.keys()))
    tf.logging.info("Read ACR articles metadata: {}".format(len(articles_metadata_df)))
    tf.logging.info("Read ACR article content embeddings: {}".format(content_article_embeddings.shape))

    return acr_label_encoders, articles_metadata_df, content_article_embeddings



def load_nar_module_preprocessing_resources(nar_module_preprocessing_resources_path):
    #{'nar_label_encoders', 'nar_standard_scalers'}
    nar_resources = \
              deserialize(nar_module_preprocessing_resources_path)

    nar_label_encoders = nar_resources['nar_label_encoders']
    tf.logging.info("Read NAR label encoders for: {}".format(nar_label_encoders.keys()))

    return nar_label_encoders    

def save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, output_dir, 
                                    training_hours_for_each_eval,
                                    output_csv='eval_stats_benchmarks.csv'):
    metrics_df = pd.DataFrame(eval_sessions_metrics_log)
    metrics_df = metrics_df.reset_index()
    metrics_df['hour'] = metrics_df['index'].apply(lambda x: ((x+1)*training_hours_for_each_eval)%24)
    metrics_df['day'] = metrics_df['index'].apply(lambda x: int(((x+1)*training_hours_for_each_eval)/24))
    
    csv_output_path = os.path.join(output_dir, output_csv)
    metrics_df.to_csv(csv_output_path, index=False)        

def upload_model_output_to_gcs(local_dir_path, gcs_model_dir, files_pattern=None):
    re_search = re.search(r'gs://([a-z0-9_]+)/', gcs_model_dir)
    if re_search:
        #Removing bucket prefix
        bucket_prefix = re_search.group(0)
        gcs_relative_path = gcs_model_dir.replace(bucket_prefix, '')

        bucket_name = re_search.group(1)
    else:
        raise Exception('Invalid model dir. Expected a GCS path: {}'.format(gcs_model_dir))

    tf.logging.info('Uploading model local cached output files from {} to {}'.format(local_dir_path, gcs_model_dir))
    upload_local_dir_to_gcs(local_folder_path=local_dir_path,
                            gcs_bucket=bucket_name,
                            gcs_relative_path=gcs_relative_path,
                            files_pattern=files_pattern)
    tf.logging.info('Finished uploading model output to GCS')


def dowload_model_output_from_gcs(local_dir_path, gcs_model_dir, files_pattern=None):
    re_search = re.search(r'gs://([a-z_]+)/', gcs_model_dir)
    if re_search:
        #Removing bucket prefix
        bucket_prefix = re_search.group(0)
        gcs_relative_path = gcs_model_dir.replace(bucket_prefix, '')

        bucket_name = re_search.group(1)
    else:
        raise Exception('Invalid model dir. Expected a GCS path: {}'.format(gcs_model_dir))

    tf.logging.info('Dowloading previously trained model checkpoints to local cached from {} to {}'.format(gcs_model_dir, local_dir_path))
    download_from_gcs_dir(local_folder_path=local_dir_path, 
                          gcs_bucket=bucket_name, 
                          gcs_relative_path=gcs_relative_path, 
                          files_pattern=files_pattern)
    tf.logging.info('Finished dowloading model output from GCS')    
