from absl import logging
from absl import flags
import tensorflow as tf
import numpy as np


FLAGS = flags.FLAGS


def update_pretrain_metrics_train(contrast_loss, contrast_acc, contrast_entropy,
                                  loss, logits_con, labels_con):
    """Updated pretraining metrics."""
    contrast_loss.update_state(loss)

    contrast_acc_val = tf.equal(
            tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
    contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
    contrast_acc.update_state(contrast_acc_val)

    prob_con = tf.nn.softmax(logits_con)
    entropy_con = -tf.reduce_mean(
            tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))
    contrast_entropy.update_state(entropy_con)



def update_finetune_metrics_train(supervised_loss_metric, supervised_acc_metric,
                                  supervised_msens_metric,loss, 
                                  labels, softmax):
    
    supervised_loss_metric.update_state(loss)
    supervised_msens_metric.update_state(labels, softmax)

    label_acc = tf.equal(tf.argmax(labels, 1), tf.argmax(softmax, axis=1))
    label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
    supervised_acc_metric.update_state(label_acc)

def allreduce_global_metrics(dist,global_loss, global_acc, global_msens,
                             local_loss , local_acc , local_msens):
    if FLAGS.distrib_library == "horovod":       
        global_loss.update_state(dist.allreduce(local_loss.result()))
        global_acc.update_state(dist.allreduce(local_acc.result()))
        global_msens.update_state(dist.allreduce(local_msens.result())) 
    elif FLAGS.distrib_library == "smdistributed":
        global_loss.update_state(dist.allreduce(local_loss.result(),0,1))
        global_acc.update_state(dist.allreduce(local_acc.result(),0,1))
        global_msens.update_state(dist.allreduce(local_msens.result(),0,1))  

    
    
        

def _float_metric_value(metric):
    """Gets the value of a float-value keras metric."""
    return metric.result().numpy().astype(float)


def log_and_write_metrics_to_summary(all_metrics, global_step):
    for metric in all_metrics:
        metric_value = _float_metric_value(metric)
        print(f'Step: [{global_step}] {metric.name} = {metric_value}')
        tf.summary.scalar(metric.name, metric_value, step=global_step)
        
        
        

class MeanSensitivity(tf.keras.metrics.Metric):
    """ Create a custom keras.metrics class to track mean sensitivity
      """
    def __init__(self, name="Mean_sensitivity", **kwargs):
        super(MeanSensitivity, self).__init__(name=name, **kwargs)
        self.num_classes = len(FLAGS.class_grouping)
        self.cm = self.add_weight("cm",(self.num_classes,self.num_classes), initializer="zeros",dtype=tf.dtypes.float32)
        

    def update_state(self, y_true, y_pred, sample_weight=None):
        update_cm = self.confusion_matrix(y_true,y_pred)
        self.cm.assign_add(update_cm)

    def result(self):
        return self.process_confusion_matrix()

    def confusion_matrix(self,y_true, y_pred):
        """
        Creates confusion matrix
        """
        y_true  = tf.argmax(y_true,1)
        y_pred = tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,num_classes=self.num_classes,dtype=tf.float32)
        return cm

    def process_confusion_matrix(self):
        """
        cm =   true A    [[0    0    0]
               true B     [0    0    1]
               true C     [0    0    1]]
                      pred A    B    c

        Returns
        -------
        mean sensitivity

        """
        placeholder = tf.zeros((self.num_classes,) ,dtype = tf.float32, name="placeholder")

        it_non_zero = 0
        for ix_true in range(self.num_classes):

            mask = np.array([0] * self.num_classes)
            mask[ix_true] = 1
            mask_tf = tf.constant(mask,dtype = tf.float32)

            if tf.not_equal(tf.reduce_sum(self.cm[ix_true,:]),tf.constant(0,dtype = tf.float32)):
                it_non_zero += 1
                sens_class = tf.ones((self.num_classes,),dtype = tf.float32) * self.cm[ix_true,ix_true] / (tf.reduce_sum(self.cm[ix_true,:]))

                placeholder = sens_class* mask_tf + placeholder * (1-mask_tf)

        mean_sens = tf.reduce_sum(placeholder) / tf.cast(it_non_zero,dtype = tf.float32)
        return mean_sens

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cm.assign(np.zeros(((self.num_classes,self.num_classes))))

