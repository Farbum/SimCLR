import math
from absl import app
from absl import flags
import tarfile
import subprocess
import sys
import os
import json
import simclr_config

FLAGS = flags.FLAGS

# in case we want to run evaluation as a processing job
#def install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])



def main(argv):
    print(FLAGS.data_dir)
    
    if FLAGS.run_mode == "aws":
#        install("tensorflow==2.4.1")   if processing job
#        output_dir = "/opt/ml/processing/evaluation" if processing job
        output_dir = "opt/ml/model/evaluation"
    else:
        output_dir = "../local_test/evaluation"
    
    import tensorflow as tf
    import simclr_model as model_lib
    import simclr_metrics as metrics
    import simclr_objective as obj_lib
    import simclr_data as data_lib
    import simclr_aug_util as aug_util
    
    
    #Extract tar.gz model file
    tar_file_path = FLAGS.model_eval_dir + "/model.tar.gz"
    tar = tarfile.open(tar_file_path, "r:gz")
    tar.extractall(FLAGS.model_eval_dir)
    tar.close()
    
    model = tf.keras.models.load_model(FLAGS.model_eval_dir + "/model.h5")
    # Check its architecture
    model.summary()

    model.trainable = False

    print(f"{len(model.trainable_variables)} trainable variables")
    print(f"{tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in model.trainable_variables])} trainable parameters")
 
    
    # Build metrics.
    all_metrics = [] 
    eval_loss_metric = tf.keras.metrics.Mean('val/supervised_loss')
    eval_acc_metric = tf.keras.metrics.Mean('val/supervised_acc')
    eval_msens_metric = metrics.MeanSensitivity('val/supervised_msens')
    all_metrics.extend([eval_loss_metric,eval_acc_metric,eval_msens_metric])
        
    


    def on_eval_single_step(features, labels):
        supervised_head_outputs = model(features, training=False)
        val_l = labels['labels']
        val_sup_loss = obj_lib.add_supervised_loss(labels=val_l, softmax=supervised_head_outputs)
        metrics.update_finetune_metrics_train(eval_loss_metric,
                                              eval_acc_metric, 
                                              eval_msens_metric,
                                              val_sup_loss,val_l, supervised_head_outputs)

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
        
        
    
    print('Starting Evaluation')


    ds_builder_eval = data_lib.dataset_builder(FLAGS.data_dir,create_val_split = True, phase = 'evaluation')
    _, ds_eval = ds_builder_eval.build_dataset(FLAGS.train_batch_size, is_training = False, seed=FLAGS.seed)

    num_val_data = ds_builder_eval.num_eval_examples
    

    eval_steps_fine = int(math.ceil(num_val_data / FLAGS.eval_batch_size))

    
    perform_online_eval(ds_eval, eval_steps_fine)
            
    report_dict = {}
    for metric in all_metrics:
        metric_value = metric.result().numpy().astype(float)
        print(f'{metric.name} = {metric_value}')
        report_dict[metric.name] = {"value": metric_value}
        metric.reset_states()


    os.makedirs(output_dir,exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        


if __name__ == '__main__':
    app.run(main)