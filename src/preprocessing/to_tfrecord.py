import tensorflow as tf

# helper functions to create features
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tfrecord(data_path, destination_path):
    """

    Args:
        data_path: where we have data organized in different folders for each class
        destination_path: where we want tfrecord files to be written to

    Returns:

    """
    raise NotImplementedError('TF record is not implemented.')
