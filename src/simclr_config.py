from absl import flags
import os

FLAGS = flags.FLAGS


flags.DEFINE_float(
    'learning_rate', 0.3,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_float(
    'learning_rate_finetuning', 0.001,
    'Initial learning rate for finetuning')

flags.DEFINE_enum(
    'learning_rate_scaling', 'linear', ['linear', 'sqrt'],
    'How to scale the learning rate as a function of batch size.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float(
    'weight_decay', 1e-6, 
    'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'train_batch_size', 32,
    'Batch size for training.')

flags.DEFINE_enum(
    'base_weights', 'None',['imagenet','None'],
    'How to initialize the modelbackbone weights')

flags.DEFINE_enum(
    'finetune_freeze', 'None',['backbone','None'],
    'Whether to freeze backbone or not for finetuning')

flags.DEFINE_string(
    'train_split', 'train',
    'Split for training.')

flags.DEFINE_integer(
    'train_epochs', 4,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'train_epochs_finetune', 4,
    'Number of epochs to finetune for.')

flags.DEFINE_bool(
    'use_step_decay', True,
    'Whether to use step decay for the finetuning learning rate')

flags.DEFINE_enum(
    'distrib_library', 'None', ['None', 'horovod', 'smdistributed'],
    'Library of choice of distributed training')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 0,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_string(
    'eval_split', 'validation',
    'Split for evaluation.')


flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune','pretrain_then_finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_bool('pretrain_on_train_only', False,
                  'Whether to pretrain only on the training set')


flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')


flags.DEFINE_enum(
    'optimizer', 'adam', ['momentum', 'adam', 'lars'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')



flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 1,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', 0,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')


flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')

flags.DEFINE_boolean(
    'use_blur', True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.')


flags.DEFINE_string(
    'data_dir', os.environ.get('SM_CHANNEL_TRAIN'),
    'Directory where dataset is stored.')

flags.DEFINE_string(
    'model_eval_dir', os.environ.get('SM_CHANNEL_MODEL_TO_EVAL'),
    'Model directory for evaluation')

flags.DEFINE_list(
    'hparams_header', [None],
    'list of hparams to display as header in TB, common to all exp. under logs/')

flags.DEFINE_list(
    'class_grouping', [["AK"],["BCC"],["BKL"],["DF"],["MEL"],["NV"],["SCC"],["VASC"]],
    'Class grouping')

flags.DEFINE_string(
    's3_tensorboard_uri', None,
    'Model directory for training.')

flags.DEFINE_boolean(
    'oversample', True,
    'Whether to oversample minority classes during pretraining')

flags.DEFINE_boolean(
    'oversample_f', True,
    'Whether to oversample minority classes during finetuning')

flags.DEFINE_boolean(
    'use_old_data_aug', False,
    'Whether to use old data augmentation/preprocessing pipeline')

flags.DEFINE_boolean(
    'aug_cutout', False,
    'Whether to use use cutout in the new augmentation pipeline')

flags.DEFINE_boolean(
    'use_dropout', False,
    'Whether to use use a dropout layer in the classification head')

flags.DEFINE_integer(
    'weight_AK', 15,
    'Oversampling weight for given class')

flags.DEFINE_integer(
    'weight_BCC', 4,
    'Oversampling weight for given class')

flags.DEFINE_integer(
    'weight_BKL', 5,
    'Oversampling weight for given class')

flags.DEFINE_integer(
    'weight_DF', 53,
    'Oversampling weight for given class')

flags.DEFINE_integer(
    'weight_MEL', 3,
    'Oversampling weight for given class')

flags.DEFINE_integer(
    'weight_NV', 1,
    'Oversampling weight for given class')

flags.DEFINE_integer(
    'weight_SCC', 20,
    'Oversampling weight for given class')

flags.DEFINE_integer(
    'weight_VASC', 50,
    'Oversampling weight for given class')

flags.DEFINE_float(
    'val_split', 0.2,
    'fraction of data to leave out for the validation set')

flags.DEFINE_enum(
    'run_mode', 'aws', ['aws', 'local'],
    'Whether to run locally or in the cloud')

flags.DEFINE_float(
    'val_center_crop_proportion', 0.875,
    'Proportion of image to retain along the less-cropped side for val center crop')

flags.DEFINE_string(
    'img_ext','jp*',
    'Dataset image extensions')

flags.DEFINE_integer(
    'seed', None,
    'Seed used to split the training and validation set')