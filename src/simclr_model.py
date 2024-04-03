import tensorflow as tf
import simclr_aug_util as aug_util
import resnet as resnet
import simclr_lars_opt as lars_optimizer
from absl import flags
import math

FLAGS = flags.FLAGS
BATCH_NORM_EPSILON = 1e-5

def get_train_steps(num_examples):
    """Determine the number of training steps."""
    return  (num_examples * FLAGS.train_epochs // FLAGS.train_batch_size) + 1


def build_optimizer(learning_rate):
    """Returns the optimizer."""
    if FLAGS.optimizer == 'momentum':
        return tf.keras.optimizers.SGD(learning_rate, FLAGS.momentum, nesterov=True)
    elif FLAGS.optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate)
    elif FLAGS.optimizer == 'lars':
        return lars_optimizer.LARSOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            weight_decay=FLAGS.weight_decay,
            exclude_from_weight_decay=[
                'batch_normalization', 'bias', 'head_supervised'
            ])
    else:
        raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))

def add_weight_decay(model, adjust_per_optimizer=True):
    """Compute weight decay from flags."""
    if adjust_per_optimizer and 'lars' in FLAGS.optimizer:
        # Weight decay are taking care of by optimizer for these cases.
        # Except for supervised head, which will be added here.
        l2_losses = [
            tf.nn.l2_loss(v)
            for v in model.trainable_variables
            if 'head_supervised' in v.name and 'bias' not in v.name
        ]
        if l2_losses:
            return FLAGS.weight_decay * tf.add_n(l2_losses)
        else:
            return 0
    
    # TODO(google): Think of a way to avoid name-based filtering here.
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_weights
        if 'batch_normalization' not in v.name
    ]
    loss = FLAGS.weight_decay * tf.add_n(l2_losses)
    return loss


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, base_learning_rate, num_examples, name=None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self._name = name
    
    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            warmup_steps = int(
                round(FLAGS.warmup_epochs * self.num_examples //
                      FLAGS.train_batch_size))
            if FLAGS.learning_rate_scaling == 'linear':
                scaled_lr = self.base_learning_rate * FLAGS.train_batch_size / 256.
            elif FLAGS.learning_rate_scaling == 'sqrt':
                scaled_lr = self.base_learning_rate * math.sqrt(FLAGS.train_batch_size)
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                    FLAGS.learning_rate_scaling))
            learning_rate = (
                step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)
        
            # Cosine decay learning rate schedule
            total_steps = get_train_steps(self.num_examples)
            # TODO(srbs): Cache this object.
            cosine_decay = tf.keras.experimental.CosineDecay(
                scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                     cosine_decay(step - warmup_steps))
      
        return learning_rate



class step_decay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies step decay schedule WITHOUT warmup."""

    def __init__(self, base_learning_rate, num_examples, threshold_ep = 10, name=None):
        super(step_decay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self.threshold_ep = threshold_ep
        self._name = name
    
    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            
            threshold_steps = int(round(self.threshold_ep* self.num_examples //
                      FLAGS.train_batch_size))

            num_decay_to_apply = step  // threshold_steps
            decay_rate = 10**(1/3)   # divides LR by 10 every 3 thresholds
            learning_rate = self.base_learning_rate / (decay_rate**num_decay_to_apply) 
        
        LR = learning_rate if FLAGS.use_step_decay else self.base_learning_rate
        return LR



def get_config(self):
    return {
        'base_learning_rate': self.base_learning_rate,
        'num_examples': self.num_examples,
    }



def resnet50v2_model(img_size=(224, 224, 3)):
    """
    builds a model on top of imagenet pre-trained resenet50v2
    Args:
        img_size (1x3 tuple): size of input images. defaults to (224, 224, 3)
        num_classes (int): number of classes, defaults to 8

    Returns:
        tf.Keras.Model: Based on ResNet50v2.
    """    
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    
    if FLAGS.base_weights == 'None':
        base_weights = None
        print("Initializing model with random weights")
    else:
        base_weights = FLAGS.base_weights
        print(f"Initializing model with {FLAGS.base_weights} weights")
        

    
    base_resnet = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=base_weights,
        input_shape=img_size,
        pooling=None
    )
    
    base_resnet.trainable = True
    
    inputs = tf.keras.Input(shape=(img_size[0],img_size[1],img_size[2]*2))    
    num_transforms = 2
    # Split channels, and optionally apply extra batched augmentation.
    features_list = tf.split(
        inputs, num_or_size_splits=num_transforms, axis=-1, name="split_last_channel")
        
    if FLAGS.use_blur:
        features_list = aug_util.batch_random_blur(features_list,
                                                    FLAGS.image_size,
                                                    FLAGS.image_size)
    features = tf.concat(features_list, 0,name="concat_back")     
    
    x = preprocess_input(features)
    x = base_resnet(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    
    #PROJECTION HEAD
    proj_head_list = [x]
    
    for i in range(FLAGS.num_proj_layers - 1):
        p_dense = tf.keras.layers.Dense(units = x.shape[-1], 
                                  activation = 'relu',
                                  use_bias=False,
                                  name=f'proj_dense_{i}')(proj_head_list[-1])
        
        p_bn = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=FLAGS.batch_norm_decay,
                epsilon=BATCH_NORM_EPSILON,
                name = f'proj_bn_{i}')(p_dense)
        
        proj_head_list.append(p_bn)
    
    p_dense_last = tf.keras.layers.Dense(units = FLAGS.proj_out_dim, 
                              activation = None,
                              use_bias=False,
                              name='proj_dense_last')(proj_head_list[-1])
    
    p_bn_last = tf.keras.layers.BatchNormalization(
                                axis=-1,
                                momentum=FLAGS.batch_norm_decay,
                                epsilon=BATCH_NORM_EPSILON,
                                name = 'proj_bn_last')(p_dense_last)
    proj_head_list.append(p_bn_last)
    proj_head_outputs = tf.identity(proj_head_list[-1], 'proj_head_output')
    
    return tf.keras.Model(inputs, proj_head_outputs)
        


def model_finetune(backbone_model, img_size=(224, 224, 3)):
    num_classes = len(FLAGS.class_grouping)
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    
    inputs = tf.keras.Input(shape=img_size)
    x = preprocess_input(inputs)
    
    start_ix = [ix for ix,layer in enumerate(backbone_model.layers) \
     if layer.name.startswith("resnet50v2")][0]
    
    if FLAGS.ft_proj_selector == 0:
        last_l_name = "global_average"
    elif FLAGS.ft_proj_selector == FLAGS.num_proj_layers:
        last_l_name = "proj_bn_last"
    else:
        last_l_name = "proj_bn_" + str(FLAGS.ft_proj_selector -1)
    
    end_ix = [ix for ix,layer in enumerate(backbone_model.layers) \
     if layer.name.startswith(last_l_name)][0]
    
    for layer in backbone_model.layers[(start_ix):(end_ix+1)]:
        layer.trainable = True
        x = layer(x)
    if FLAGS.use_dropout:
        if FLAGS.ft_proj_selector == 0:
            x = tf.keras.layers.Dropout(.75)(x)
        else:
            x = tf.keras.layers.Dropout(.5)(x)

    supervised_head_outputs = tf.keras.layers.Dense(num_classes, 
                                                    activation="softmax",
                                                    name='sup_dense')(x)    
    
    return tf.keras.Model(inputs, supervised_head_outputs)
