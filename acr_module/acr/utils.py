import tensorflow as tf
import pickle
from time import time

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

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]        


def resolve_files(regex):
    """Return list of files given a regex"""
    list_op = tf.train.match_filenames_once(regex)
    init_ops = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_ops)
        files = sess.run(list_op)

    return list(sorted(files))        