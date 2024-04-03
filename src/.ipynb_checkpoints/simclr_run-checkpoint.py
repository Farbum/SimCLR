import json
import math
import os
import datetime
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import smdistributed.dataparallel.tensorflow as sdp
import boto3

import simclr_model as model_lib
import simclr_config
import simclr_metrics as metrics
import simclr_objective as obj_lib
import simclr_data as data_lib
import simclr_aug_util as aug_util


FLAGS = flags.FLAGS




def json_serializable(val):
    try:
        json.dumps(val)
        return True
    except TypeError:
        return False

    
def check_TB_parent_event(TB_log_dir):
    """
    Check if we already have a parent event file in the S3 path for Tensorboard.
    Returns:
    empty: False if there's an event file in this path. True otherwise.
    """
    s3_client = boto3.client('s3')

    bucket = TB_log_dir.split('/')[2]
    prefix = '/'.join(TB_log_dir.split('/')[3:-1])
    print(f"LOOKING in bucket {bucket} with prefix {prefix} for TENSORBOARD PARENT FILE")

    resp = s3_client.list_objects(
        Bucket=bucket,
        Prefix=prefix
    )
    is_empty = len([x['Key'] for x in resp['Contents'] if x['Key'].startswith(os.path.join(prefix, 'events.out.tfevents'))])==0
    return is_empty    
    

def TB_log_hparams(sm_job_name):
    """
    Logs hyperparameters and metrics of an specific experiment to the Hparams tab in Tensorboard.
    """
    def format_value(v):
        return v if type(v) in [bool, int, float, str] else str(v)
    
    hparams = {k:format_value(FLAGS.flag_values_dict()[k]) for k in FLAGS.hparams_header}
    hp.hparams(hparams, trial_id = sm_job_name)
        
    
def get_salient_tensors_dict():
    """Returns a dictionary of tensors."""
    graph = tf.compat.v1.get_default_graph()
    result = {}
    for i in range(1, 5):
        result['block_group%d' % i] = graph.get_tensor_by_name(
            'resnet/block_group%d/block_group%d:0' % (i, i))
    result['initial_conv'] = graph.get_tensor_by_name(
        'resnet/initial_conv/Identity:0')
    result['initial_max_pool'] = graph.get_tensor_by_name(
        'resnet/initial_max_pool/Identity:0')
    result['final_avg_pool'] = graph.get_tensor_by_name('resnet/final_avg_pool:0')
    result['logits_sup'] = graph.get_tensor_by_name(
        'head_supervised/logits_sup:0')
    
    return result

class best_model_saver():
    def __init__(self):
        self.best_val_msens = 0.
        self.last_saved_model_name = ""
        self.dir_to_save = self.get_dir_to_save()
        
    def get_dir_to_save(self):
        if FLAGS.run_mode =="aws":      
            dir_to_save = "/opt/ml/model/"
        else:
            dir_to_save = '../local_test/model_dir/save_models/'
        return dir_to_save
    
    def compare_and_save(self,model,val_msens):
        if val_msens > self.best_val_msens:

            #removing last saved model
            if self.best_val_msens > 0.:
                os.remove(self.dir_to_save + self.last_saved_model_name) 
            #saving model
            model_name = "best_model_" + str(val_msens)[:5] + ".h5"
            model.save(self.dir_to_save + model_name)
            self.last_saved_model_name = model_name
            self.best_val_msens = val_msens
            

def build_saved_model(model):
    """Returns a tf.Module for saving to SavedModel."""

    class SimCLRModel(tf.Module):
        """Saved model for exporting to hub."""
      
        def __init__(self, model):
            self.model = model
            # This can't be called `trainable_variables` because `tf.Module` has
            # a getter with the same name.
            self.trainable_variables_list = model.trainable_variables
      
        @tf.function
        def __call__(self, inputs, trainable):
            self.model(inputs, training=trainable)
            return get_salient_tensors_dict()
    
    module = SimCLRModel(model)
    input_spec = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)
    module.__call__.get_concrete_function(input_spec, trainable=True)
    module.__call__.get_concrete_function(input_spec, trainable=False)
    return module



def try_restore_from_checkpoint(model, global_step, optimizer):
    """Restores the latest ckpt if it exists, otherwise check FLAGS.checkpoint."""
    checkpoint = tf.train.Checkpoint(
        model=model, global_step=global_step, optimizer=optimizer)
    
    if FLAGS.run_mode == 'local':
        ckpt_dir = '../local_test/model_dir/'
    elif FLAGS.run_mode == 'aws':
        ckpt_dir = "/opt/ml/checkpoints/"
        
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=ckpt_dir,
        max_to_keep=FLAGS.keep_checkpoint_max)
    latest_ckpt = checkpoint_manager.latest_checkpoint
    if latest_ckpt:
        # Restore model weights, global step, optimizer states
        logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
        checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
    elif FLAGS.checkpoint:
        # Restore model weights only, but not global step and optimizer states
        logging.info('Restoring from given checkpoint: %s', FLAGS.checkpoint)
        checkpoint_manager2 = tf.train.CheckpointManager(
            tf.train.Checkpoint(model=model),
            directory=ckpt_dir,
            max_to_keep=FLAGS.keep_checkpoint_max)
        checkpoint_manager2.checkpoint.restore(FLAGS.checkpoint).expect_partial()
        if FLAGS.zero_init_logits_layer:
            model = checkpoint_manager2.checkpoint.model
            output_layer_parameters = model.supervised_head.trainable_weights
            logging.info('Initializing output layer parameters %s to zero',
                         [x.op.name for x in output_layer_parameters])
            for x in output_layer_parameters:
              x.assign(tf.zeros_like(x))
    
    return checkpoint_manager









def main(argv):
    print(FLAGS.data_dir)

    model = model_lib.resnet50v2_model(img_size=(224, 224, 3))
    model.summary()

    #Initializing data parallel library's client
    sdp.init()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[sdp.local_rank()], 'GPU')
        
    #Creating Tensorboard summary writer
    if FLAGS.run_mode == "aws":
        summary_writer = tf.summary.create_file_writer(FLAGS.s3_tensorboard_uri)
        if check_TB_parent_event(FLAGS.s3_tensorboard_uri):
            print("Writing Tensorboard parent event file")
            tensorboard_parent_file_writer = tf.summary.create_file_writer('/'.join(FLAGS.s3_tensorboard_uri.split("/")[:-1]) + "/")
            with tensorboard_parent_file_writer.as_default():
                hp.hparams_config(
                    hparams=[hp.HParam(h, display_name=h) for h in FLAGS.hparams_header],
                    metrics=[hp.Metric('train/contrast_loss',display_name = 'train/contrast_loss'),
                             hp.Metric('train/supervised_msens',display_name = 'train/supervised_msens'),
                             hp.Metric('val/supervised_msens',display_name = 'val/supervised_msens')]
                )
        # Logging experiment hyperparameters for TB
        print("logging hyperparameters")
        with summary_writer.as_default():
            TB_log_hparams(sm_job_name = FLAGS.s3_tensorboard_uri.split("/")[-1])
            
    elif FLAGS.run_mode == 'local':
        dt = str(datetime.datetime.now())[:19]
        local_log_dir = '../local_test/TB_log/' + dt
        summary_writer = tf.summary.create_file_writer(local_log_dir)
    
    


    # Build metrics.
    all_metrics = []  # For summaries.
    weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
    all_metrics.extend([weight_decay_metric])
    if 'pretrain' in FLAGS.train_mode:
        pretrain_total_loss_metric = tf.keras.metrics.Mean('train/pretrain_total_loss')
        contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
        contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
        contrast_entropy_metric = tf.keras.metrics.Mean('train/contrast_entropy')
        all_metrics.extend([
                contrast_loss_metric, contrast_acc_metric, 
                contrast_entropy_metric,pretrain_total_loss_metric
        ])
    if 'finetune' in FLAGS.train_mode or (FLAGS.train_suphead_while_pretraining and FLAGS.pretrain_on_train_only):
        supervised_total_loss_metric = tf.keras.metrics.Mean('train/total_supervised_loss')
        supervised_global_loss_metric = tf.keras.metrics.Mean('train/global_total_supervised_loss')
        supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
        supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')
        supervised_msens_metric = metrics.MeanSensitivity('train/supervised_msens')
        v_supervised_loss_metric = tf.keras.metrics.Mean('val/supervised_loss')
        v_supervised_acc_metric = tf.keras.metrics.Mean('val/supervised_acc')
        v_supervised_msens_metric = metrics.MeanSensitivity('val/supervised_msens')
        regularization_loss    = tf.keras.metrics.Mean('val/regularization_loss')
        eval_metrics = [v_supervised_loss_metric, v_supervised_acc_metric, v_supervised_msens_metric,regularization_loss]
        all_metrics.extend([supervised_total_loss_metric,supervised_global_loss_metric,
                            supervised_loss_metric, supervised_acc_metric,supervised_msens_metric])
        all_metrics.extend(eval_metrics)
        
        
    # Build graph for online eval of validation set
    if 'finetune' in FLAGS.train_mode:
        
        def on_eval_single_step(features, labels):
            supervised_head_outputs = f_model(features, training=False)
            val_l = labels['labels']
            val_sup_loss = obj_lib.add_supervised_loss(labels=val_l, softmax=supervised_head_outputs)
            metrics.update_finetune_metrics_train(v_supervised_loss_metric,
                                                  v_supervised_acc_metric, 
                                                  v_supervised_msens_metric,
                                                  val_sup_loss,val_l, supervised_head_outputs)
            reg_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True)
            regularization_loss.update_state(reg_loss) 
    
        @tf.function
        def on_eval_run_single_step(iterator):
            images, labels = next(iterator)
            features, labels = images, {'labels': labels}
            on_eval_single_step(features, labels)    
            
        def perform_online_eval(ds, eval_steps):
            """Perform online evaluation on validation set."""
            print(f'starting evaluation on validation set, {eval_steps} steps to go')
            iterator = iter(ds)
            for i in range(eval_steps):
                on_eval_run_single_step(iterator)
            print('Evaluation step completed')
        
        


                
    
    
    if 'pretrain' in FLAGS.train_mode :
    
        ds_builder = data_lib.dataset_builder(FLAGS.data_dir, create_val_split = FLAGS.pretrain_on_train_only, 
                                              phase = 'pretrain')
        num_train_examples = ds_builder.num_train_examples
        num_eval_examples = ds_builder.num_eval_examples
        
        
        train_steps = model_lib.get_train_steps(num_train_examples)
        epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))
    
        print(f'train examples: {num_train_examples}')
        print(f'train_steps: {train_steps}')
        print(f'eval examples: {num_eval_examples}')
        
        
        checkpoint_steps = (
                FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * epoch_steps))

        steps_per_loop = checkpoint_steps
        
        ds_tr, ds_val = ds_builder.build_dataset(FLAGS.train_batch_size, is_training = True, seed=FLAGS.seed)
        learning_rate = model_lib.WarmUpAndCosineDecay(FLAGS.learning_rate,
                                                   num_train_examples)
        p_optimizer = model_lib.build_optimizer(learning_rate)
        
        # Restore checkpoint if available.
        checkpoint_manager = try_restore_from_checkpoint(
            model, p_optimizer.iterations, p_optimizer)
        
        print(f"{len(model.trainable_variables)} trainable layers for pretraining")
        print(f"{tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in model.trainable_variables])}\
              trainable parameters for pretraining")
        
        
        pretrain_step = p_optimizer.iterations
        cur_step = pretrain_step.numpy()
        iterator = iter(ds_tr)
        

        
        def pretrain_single_step(features, labels,im_ixes):
            with tf.GradientTape() as tape:
    
                should_record = tf.equal((p_optimizer.iterations + 1) % steps_per_loop, 0)
                with tf.summary.record_if(should_record):
                    # Only log augmented images for the first tower.
                    tf.summary.image(
                            'image', features[:, :, :, :3], step=p_optimizer.iterations + 1)
                    
                    
                projection_head_outputs = model(features, training=True)
    
                loss = None
                con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
                        projection_head_outputs,
                        im_ixes,
                        hidden_norm=FLAGS.hidden_norm,
                        temperature=FLAGS.temperature,
                        strategy=None)
                if loss is None:
                    loss = con_loss
                else:
                    loss += con_loss
                metrics.update_pretrain_metrics_train(contrast_loss_metric,
                                                        contrast_acc_metric,
                                                        contrast_entropy_metric,
                                                        con_loss, logits_con,
                                                        labels_con)
    
                weight_decay = model_lib.add_weight_decay(
                        model, adjust_per_optimizer=True)
                weight_decay_metric.update_state(weight_decay)
                loss += weight_decay
                pretrain_total_loss_metric.update_state(loss)
    
                grads = tape.gradient(loss, model.trainable_variables)
                p_optimizer.apply_gradients(zip(grads, model.trainable_variables))

    
        @tf.function
        def pretrain_multiple_steps(iterator):
            # `tf.range` is needed so that this runs in a `tf.while_loop` and is
            # not unrolled.
            for _ in tf.range(steps_per_loop):
                with tf.name_scope(''):
                    images, labels, im_ixes = next(iterator)
                    features, labels = images, {'labels': labels}
                    #strategy.run(single_step, (features, labels))
                    pretrain_single_step(features,labels, im_ixes)
        

        while cur_step < train_steps:

            with summary_writer.as_default():
                pretrain_multiple_steps(iterator)
                cur_step = pretrain_step.numpy()
                checkpoint_manager.save(cur_step)
                print(f'Completed training: {cur_step} / {train_steps} steps')
                tf.summary.scalar('learning_rate',
                        learning_rate(tf.cast(pretrain_step, dtype=tf.float32)),
                        pretrain_step)
                
            #logging metrics after online_eval to allow val/metrics computation and update
            with summary_writer.as_default(): 
                metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
                summary_writer.flush()
            last_closs = contrast_loss_metric.result().numpy().astype(float)
            
            for metric in all_metrics:
                metric.reset_states()

        model_name = "pretrain_model_" + str(last_closs)[:5] + ".h5"
        if FLAGS.run_mode =="aws":            
            model.save("/opt/ml/model/" + model_name)
        else:
            model.save('../local_test/model_dir/save_models/' + model_name)
        pretrain_final_step = pretrain_step.numpy()
        print('Training complete...')

 







   
    
    if 'finetune' in FLAGS.train_mode:
        print('Starting Finetuning')
        
        print('Attaching classification head to backbone')
        f_model = model_lib.model_finetune(model)
        f_model.summary()
        

        
        
        # Freezing backbone layers if needed
        if FLAGS.finetune_freeze == "backbone":
            print("Freezing backbone layers for finetuning")
            for layer in f_model.layers:
                if layer.name == 'resnet50v2' or 'proj_' in layer.name:
                    layer.trainable = False
            print(f"{len(f_model.trainable_variables)} trainable variables after freezing backbone")
            print(f"{tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in f_model.trainable_variables])} trainable parameters after freezing backbone")
        
        print("List of trainable layers for finetuning:\n"+"\n".join(['   '+v.name for v in f_model.trainable_variables]))
        
                
        
        
        ds_builder_finetune = data_lib.dataset_builder(FLAGS.data_dir,create_val_split = True, phase = 'finetune')
        ds_tr_f, ds_val_f = ds_builder_finetune.build_dataset(FLAGS.train_batch_size, is_training = True, seed=FLAGS.seed)
        
        num_tr_data  = ds_builder_finetune.num_train_examples
        num_val_data = ds_builder_finetune.num_eval_examples
        
        f_learning_rate = model_lib.step_decay(FLAGS.learning_rate_finetuning,
                                               num_tr_data)
        
        f_learning_rate = f_learning_rate * sdp.size()
        
        f_optimizer = model_lib.build_optimizer(f_learning_rate)
        
        if not'pretrain' in FLAGS.train_mode:
            sdp.broadcast_variables(f_model.variables, root_rank=0)
            sdp.broadcast_variables(f_optimizer.variables(), root_rank=0)

        train_steps_fine =  num_tr_data * FLAGS.train_epochs_finetune // FLAGS.train_batch_size + 1
        eval_steps_fine = int(math.ceil(num_val_data / FLAGS.eval_batch_size))
        epoch_steps_fine = int(round(num_tr_data / FLAGS.train_batch_size))
    
        print(f'train examples: {num_tr_data}')
        print(f'train_steps: {train_steps_fine}')
        print(f'eval examples: {num_val_data}')
        print(f'eval steps: {eval_steps_fine}')
        
        if not'pretrain' in FLAGS.train_mode:
            # Restore checkpoint if available.
            pretrain_final_step = 0
            checkpoint_manager = try_restore_from_checkpoint(
                f_model, f_optimizer.iterations, f_optimizer)
            
        model_saver = best_model_saver()
        
        checkpoint_steps = FLAGS.checkpoint_epochs * epoch_steps_fine
        steps_per_loop = checkpoint_steps
        
        fine_step = f_optimizer.iterations
        cur_fine_step = fine_step.numpy()
        f_iterator = iter(ds_tr_f)
        
        def finetune_single_step(features, labels):
            with tf.GradientTape() as tape:
    
                should_record = tf.equal((f_optimizer.iterations + 1) % steps_per_loop, 0)
                with tf.summary.record_if(should_record):
                    # Only log augmented images for the first tower.
                    tf.summary.image(
                            'image', features[:, :, :, :3], step=f_optimizer.iterations + 1)
                    
                loss = None 
                sup_head_output = f_model(features, training=True)
    
                l = labels['labels']

                sup_loss = obj_lib.add_supervised_loss(labels=l, softmax=sup_head_output)
                metrics.update_finetune_metrics_train(supervised_loss_metric,
                                      supervised_acc_metric, 
                                      supervised_msens_metric,
                                      sup_loss,l, sup_head_output)
                
                if loss is None:
                    loss = sup_loss
                else:
                    loss += sup_loss
    
                weight_decay = model_lib.add_weight_decay(
                        f_model, adjust_per_optimizer=True)
                weight_decay_metric.update_state(weight_decay)
                loss += weight_decay
                supervised_total_loss_metric.update_state(loss)
                
                global_loss = sdp.oob_allreduce(loss)
                supervised_loss_metric.update_state(global_loss)
                
                tape = sdp.DistributedGradientTape(tape)
                grads = tape.gradient(loss, f_model.trainable_variables)
                f_optimizer.apply_gradients(zip(grads, f_model.trainable_variables))

    
        @tf.function
        def finetune_multiple_steps(iterator):
            # `tf.range` is needed so that this runs in a `tf.while_loop` and is
            # not unrolled.
            for _ in tf.range(steps_per_loop):
                with tf.name_scope(''):
                    images, labels = next(iterator)
                    features, labels = images, {'labels': labels}
                    #strategy.run(single_step, (features, labels))
                    finetune_single_step(features,labels)
        while cur_fine_step < train_steps_fine:
            # Calls to tf.summary.xyz lookup the summary writer resource which is
            # set by the summary writer's context manager.
            with summary_writer.as_default():
                finetune_multiple_steps(f_iterator)
                cur_fine_step = fine_step.numpy()
                global_step = cur_fine_step + pretrain_final_step
                if sdp.rank() == 0:
                    checkpoint_manager.save(global_step)

                    print(f'Finetuning - step: {cur_fine_step} / {train_steps_fine} steps')
                    tf.summary.scalar('learning_rate',
                                      f_learning_rate(tf.cast(cur_fine_step, dtype=tf.float32)),
                                      global_step)
            
            if sdp.rank() == 0:
                #perform online validation assessment and save model if best perf
                perform_online_eval(ds_val_f, eval_steps_fine)
                model_saver.compare_and_save(f_model, v_supervised_msens_metric.result().numpy().astype(float))
            
                #logging metrics after online_eval to allow val/metrics computation and update
                with summary_writer.as_default(): 
                    metrics.log_and_write_metrics_to_summary(all_metrics, global_step)
                    summary_writer.flush()
                last_vmsens = v_supervised_msens_metric.result().numpy().astype(float)
                
            for metric in all_metrics:
                metric.reset_states()
        
        #Saving model
        if sdp.rank() == 0:
            f_model_name = "final_model_" + str(last_vmsens)[:5] + ".h5"
            if FLAGS.run_mode =="aws":
                f_model.save("/opt/ml/model/" + f_model_name)
            else:
                f_model.save('../local_test/model_dir/save_models/' + f_model_name)


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    # For outside compilation of summaries on TPU.
    tf.config.set_soft_device_placement(True)
    app.run(main)