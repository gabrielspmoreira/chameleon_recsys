import tensorflow as tf
from google.cloud import storage
import os
import glob

def get_dir_recursive_files(base_dir):
    recursive_files = glob.iglob("{}/**/*".format(base_dir), recursive=True)
    relative_paths = [path.replace(base_dir,'') for path in recursive_files if os.path.isfile(path)]
    return list(relative_paths)

def upload_local_file_to_gcs(local_file_path, gcs_bucket, gcs_relative_path):
	CHUNK_SIZE = 10485760  # 10MB
	client = storage.Client()
	bucket = client.get_bucket(gcs_bucket)
	blob = bucket.blob(gcs_relative_path, 
					   chunk_size=CHUNK_SIZE
					   )

	blob.upload_from_filename(local_file_path)


def upload_local_dir_to_gcs(local_folder_path, gcs_bucket, gcs_relative_path, files_pattern=None):
	basedir = local_folder_path+"/"
	relative_paths = get_dir_recursive_files(basedir)
	for file_relative_path in relative_paths:	
		#If there is a pattern to filter files, ignores files that do not match the pattern	
		if files_pattern != None:
			file_name = os.path.basename(file_relative_path)
			if len(list(filter(lambda p: file_name.find(p) != -1, files_pattern))) == 0:
				continue


		local_file_path = os.path.join(basedir, file_relative_path)
		print(local_file_path)
		remote_file_path = os.path.join(gcs_relative_path, file_relative_path)
		tf.logging.info('Uploading {} to gs://{}/{}'.format(file_relative_path, gcs_bucket, remote_file_path))
		
		upload_local_file_to_gcs(local_file_path=local_file_path,
								 gcs_bucket=gcs_bucket,
								 gcs_relative_path=remote_file_path)


def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        /a/1.txt
        /a/b/2.txt

    If you just specify prefix = '/a', you'll get back:

        /a/1.txt
        /a/b/2.txt

    However, if you specify prefix='/a' and delimiter='/', you'll get back:

        /a/1.txt

    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

    
    remote_files = list([blob.name for blob in blobs])
    return remote_files


def download_file_from_gcs(local_file_path, gcs_bucket, gcs_relative_path):
	client = storage.Client()
	bucket = client.get_bucket(gcs_bucket)
	blob = bucket.blob(gcs_relative_path)

	blob.download_to_filename(local_file_path)


def download_from_gcs_dir(local_folder_path, gcs_bucket, gcs_relative_path, files_pattern=None):
	gcs_full_paths = list_blobs_with_prefix(gcs_bucket, prefix=gcs_relative_path)

	for gcs_path in gcs_full_paths:	
		relative_path = gcs_path.replace(gcs_relative_path, '')
		if relative_path[0] == '/':
			relative_path = relative_path[1:]

		file_name = os.path.basename(relative_path)
		#If there is a pattern to filter files, ignores files that do not match the pattern	
		if files_pattern != None:			
			if len(list(filter(lambda p: file_name.find(p) != -1, files_pattern))) == 0:
				continue


		local_file_path = os.path.join(local_folder_path, relative_path)
		local_file_dir = os.path.dirname(local_file_path)
		if not os.path.exists(local_file_dir):
			os.makedirs(local_file_dir)

		remote_file_path = os.path.join(gcs_relative_path, relative_path)

		tf.logging.info('Downloading from gs://{}/{} to {}'.format(gcs_bucket, remote_file_path, local_file_path))
		
		download_file_from_gcs(local_file_path=local_file_path,
								 gcs_bucket=gcs_bucket,
								 gcs_relative_path=remote_file_path)	




'''
Testing commands:
upload_local_dir_to_gcs(local_folder_path='/media/gabrielpm/9c21d9de-38dd-4b87-8c4e-fbae7652ab28/projects/personal/doutorado/code/chameleon/',
						gcs_bucket='mlengine_staging',
						gcs_relative_path='test')

download_from_gcs_dir(local_folder_path='/media/gabrielpm/9c21d9de-38dd-4b87-8c4e-fbae7652ab28/projects/personal/doutorado/code/chameleon/',
						gcs_bucket='mlengine_jobs',
						gcs_relative_path='gcom/nar_module/gabrielpm_gcom_nar_2018_07_13_115421/',
						files_pattern=['.csv', 'checkpoint'])						
'''