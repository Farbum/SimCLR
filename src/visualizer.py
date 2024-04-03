import tensorflow as tf
import numpy as np

import logging
import argparse
import os
import boto3

from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import TensorBoard, Callback

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

class CustomMetricsLogger(Callback):
    def __init__(self, visualizer=None, custom_epoch_metrics=None, custom_train_metrics=None):
        self.visualizer = visualizer
        self.custom_epoch_metrics = custom_epoch_metrics
        self.custom_train_metrics = custom_train_metrics
    
    def on_epoch_end(self, epoch, logs={}):
        metrics = dict()
        metrics = {m:logs[m] for m in self.custom_epoch_metrics}
        self.visualizer.log_metrics(metrics, epoch)

    def on_train_end(self, logs={}):
        metrics = dict()
        metrics = {m:logs[m] for m in self.custom_train_metrics}
        self.visualizer.log_metrics(metrics)
            
        roc_curves = logs['roc_curves']
        for class_name in roc_curves.keys():
            fpr = [int(100*v) for v in roc_curves[class_name]['fpr']] # convert values to int %, to be used as epochs in tf.summary
            tpr = [100*v for v in roc_curves[class_name]['tpr']] # convert values to %, for compatibility with fpr
            for idx in range(len(fpr)):
                point = {os.path.join('roc_curves', class_name): tpr[idx]}
                self.visualizer.log_metrics(point, epoch=fpr[idx])
        

class Visualizer():
    
    def __init__(self, args=None, sm_job_name=None):
        self.args = args
        self.sm_job_name = sm_job_name
        self._tensorboard_file_writer = tf.summary.create_file_writer(os.path.join(args.tensorboard_log_dir, sm_job_name))
        self._custom_metrics_logger = CustomMetricsLogger(self, self.args.custom_epoch_metrics, self.args.custom_train_metrics)
        if self._check() and self.args.filter_hparams:
            self._tensorboard_parent_file_writer = tf.summary.create_file_writer(args.tensorboard_log_dir)
            self._log_hparams_header()
            
    def _check(self):
            """
            Check if we already have a parent event file in the S3 path for Tensorboard.
            
            Returns:
                empty: False if there's an event file in this path. True otherwise.
            """
            s3 = boto3.client('s3')

            bucket = self.args.tensorboard_log_dir.split('/')[2]
            prefix = '/'.join(self.args.tensorboard_log_dir.split('/')[3:])+'/'

            resp = s3.list_objects(
                Bucket=bucket,
                Prefix=prefix
            )
            empty = len([x['Key'] for x in resp['Contents'] if x['Key'].startswith(os.path.join(prefix, 'events.out.tfevents'))])==0
            return empty

    def _log_hparams_header(self):
        """
        Creates a parent event file containing hparams/metrics header in Tensorboard.
        This file is shared between experiments.
        """
        with self._tensorboard_parent_file_writer.as_default():
            hp.hparams_config(
                hparams=[hp.HParam(h, display_name=h) for h in self.args.tensorboard_hparams_header],
                metrics=[hp.Metric(h, display_name=h) for h in self.args.tensorboard_metrics_header],
            )

    def log_training(self):
        """
        Creates Keras callbacks to log metrics included in the compiled model, as well as custom metrics.
        
        Returns:
            callbacks: a list of Keras callbacks to be included when fitting the model to the data.
        """
        callbacks = list()
        callbacks.append(TensorBoard(log_dir=os.path.join(self.args.tensorboard_log_dir, self.sm_job_name), update_freq='epoch'))
        callbacks.append(self._custom_metrics_logger)
        return callbacks

    def _log_decorator(func):
        """
        A decorator that uses a file writer to write summaries and log event files produced by other methods to the S3 path for Tensorboard.
        """
        def wrapper(self, *args, **kwargs):
            with self._tensorboard_file_writer.as_default():
                func(self, *args, **kwargs)
        return wrapper

    @_log_decorator
    def log_hparams(self):
        """
        Logs hyperparameters and metrics of an specific experiment to the Hparams tab in Tensorboard.
        """
        def format_value(v):
            return v if type(v) in [bool, int, float, str] else str(v)
        if self.args.filter_hparams:
            hparams = {h:format_value(self.args.__dict__[h]) for h in self.args.tensorboard_hparams_header}
        else:
            hparams = {k:format_value(v) for k,v in self.args.__dict__.items()}
        hp.hparams(hparams, trial_id=self.sm_job_name)

    @_log_decorator
    def log_metrics(self, metrics, epoch=0):
        """
        Logs metrics of an specific experiment and epoch to the Scalars tab in Tensorboard.
        
        Arguments:
            metrics: a dict where keys are metric names while values are metric values.
            epoch: the epoch corresponding to the metrics to be logged. Optional for end of training metrics.
        """
        for k, v in metrics.items():
            tf.summary.scalar(k, v, step=epoch)

    @_log_decorator
    def log_wrong_images(self, dataset, model):
        """
        Logs images of an specific experiment for which the true and predicted labels are different to the Images tab in Tensorboard.
        
        Arguments:
            dataset: a dataset containing (image, true label, path) for each sample
            model: a trained model used to produce the predicted label for each sample
        """
        class_names = ['_'.join(c) for c in self.args.class_group]
        wrong_count = [0] * len(class_names)
        max_count = [self.args.tensorboard_max_imgs_per_class] * len(class_names)
        for sample in dataset.unbatch().shuffle(dataset.__len__()):
            if wrong_count == max_count:
                break
            x, y, path = sample[0], sample[1], sample[2].numpy().decode()
            image = tf.expand_dims(x, axis=0)
            y_pred = np.argmax(model.predict(image))
            y_true = np.argmax(y.numpy())
            if (y_pred != y_true) and (wrong_count[y_true] < self.args.tensorboard_max_imgs_per_class):
                label_pred = class_names[y_pred]
                label_true = class_names[y_true]
                img_name = path.split('/')[-1]
                tf.summary.image(os.path.join('prediction_errors_'+label_true, label_pred, img_name), image, step=0)
                wrong_count[y_true] += 1

    @_log_decorator
    def log_compared_images(self, dataset):
        """
        Logs pairs of raw and transformed images of an specific experiment to the Images tab in Tensorboard.
        
        Arguments:
            dataset: a dataset containing (raw image, transformed image, label, path) for each sample
        """
        class_names = ['_'.join(c) for c in self.args.class_group]
        wrong_count = [0] * len(class_names)
        max_count = [self.args.tensorboard_max_imgs_per_class] * len(class_names)
        for sample in dataset.unbatch().shuffle(dataset.__len__()):
            if wrong_count == max_count:
                break
            raw, transformed, y, path = sample[0], sample[1], sample[2], sample[3].numpy().decode()
            y = np.argmax(y.numpy())
            label = class_names[y]
            if wrong_count[y] < self.args.tensorboard_max_imgs_per_class:
                pair = tf.concat([raw, transformed], axis=1)
                pair = tf.expand_dims(pair, axis=0)
                img_name = path.split('/')[-1]
                tf.summary.image(os.path.join('raw_transformed_'+label, img_name), pair, step=0)
                wrong_count[y] += 1