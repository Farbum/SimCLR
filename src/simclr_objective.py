"""Contrastive loss functions."""

from absl import flags

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

LARGE_NUM = 1e9


def add_supervised_loss(labels, softmax):
    """Compute mean supervised loss over local batch."""
    losses = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)(labels,softmax)
    return tf.reduce_mean(losses)




def add_contrastive_loss(hidden,
                        im_ixes,
                        hidden_norm=True,
                        temperature=1.0,
                        strategy=None):
    """Compute loss for model.
    Args:
        hidden: hidden vector (`Tensor`) of shape (bsz, dim).
        im_ixes: unique identifier of the original images. With oversampling, each 
                 transformation of the same image will share the same identifier
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        strategy: context information for tpu.
    Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, num_or_size_splits = 2, axis = 0)
    batch_size = tf.shape(hidden1)[0]
    
    # TODO: It looks like it is more efficient to bring together batches across workers
    # Gather hidden1/hidden2 across replicas and create local labels.
    if strategy is not None:
        hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
        hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
        enlarged_batch_size = tf.shape(hidden1_large)[0]
        # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
        replica_context = tf.distribute.get_replica_context()
        replica_id = tf.cast(
                tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
        labels_idx = tf.range(batch_size) + replica_id * batch_size
        labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
        masks = tf.one_hot(labels_idx, enlarged_batch_size)
    else:
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)
        
    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature
    
    if FLAGS.oversample:
        exp_a = tf.math.exp(tf.concat([logits_ab, logits_aa], 1))
        exp_b = tf.math.exp(tf.concat([logits_ba, logits_bb], 1))
        
        #Creating 0/1 flags of same images
        im_ixes_concat = tf.concat([im_ixes, im_ixes], -1)        
        flag_soi = tf.cast(tf.equal(tf.expand_dims(im_ixes,-1), im_ixes_concat), tf.float32)
                    
        l_a = tf.reduce_sum(tf.multiply(flag_soi,exp_a),axis = 1) / tf.reduce_sum(exp_a,axis = 1)
        l_b = tf.reduce_sum(tf.multiply(flag_soi,exp_b),axis = 1) / tf.reduce_sum(exp_b,axis = 1)
        loss_a = -tf.math.log(l_a)
        #tf.print("loss_a",loss_a)
        loss_b = -tf.math.log(l_b)
        #tf.print("loss_b",loss_b)
        
    else:
        
        loss_a = tf.nn.softmax_cross_entropy_with_logits(
                labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.nn.softmax_cross_entropy_with_logits(
                labels, tf.concat([logits_ba, logits_bb], 1))
        
    loss = tf.reduce_mean(loss_a + loss_b)
    return loss, logits_ab, labels


def tpu_cross_replica_concat(tensor, strategy=None):
    """Reduce a concatenation of the `tensor` across TPU cores.
    Args:
        tensor: tensor to concatenate.
        strategy: A `tf.distribute.Strategy`. If not set, CPU execution is assumed.
    Returns:
        Tensor of the same rank as `tensor` with first dimension `num_replicas`
        times larger.
    """
    if strategy is None or strategy.num_replicas_in_sync <= 1:
        return tensor

    num_replicas = strategy.num_replicas_in_sync

    replica_context = tf.distribute.get_replica_context()
    with tf.name_scope('tpu_cross_replica_concat'):
        # This creates a tensor that is like the input tensor but has an added
        # replica dimension as the outermost dimension. On each replica it will
        # contain the local values and zeros for all other values that need to be
        # fetched from other replicas.
        ext_tensor = tf.scatter_nd(
                indices=[[replica_context.replica_id_in_sync_group]],
                updates=[tensor],
                shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0))

        # As every value is only present on one replica and 0 in all others, adding
        # them all together will result in the full tensor on all replicas.
        ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
                                                ext_tensor)

        # Flatten the replica dimension.
        # The first dimension size will be: tensor.shape[0] * num_replicas
        # Using [-1] trick to support also scalar input.
        return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])