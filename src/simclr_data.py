import functools
from absl import flags
from absl import logging
from absl import app

import simclr_aug_util as data_util
import simclr_config
import tensorflow.compat.v2 as tf
import numpy as np
import os
import time


FLAGS = flags.FLAGS


class dataset_builder():
    
    def __init__(self,root_folder, create_val_split = False, phase = 'pretrain'):
    
        self.class_grouping = FLAGS.class_grouping
        self.num_classes = len(self.class_grouping)
        self.root_folder = root_folder
        self.create_val_split = create_val_split
        self.phase = phase
        self.num_train_examples, self.num_eval_examples = self.get_num_examples()

    

    def get_num_examples(self):
        nb_images_tr  = 0
        nb_images_val = 0
        for class_list in self.class_grouping:
            for _cl in class_list:
                cl_images = len(tf.io.gfile.glob(os.path.join(self.root_folder, _cl, f'*.{FLAGS.img_ext}')))
                if not(self.create_val_split):
                    nb_images_tr += cl_images
                else:
                    nb_images_tr += int((1 - FLAGS.val_split) * cl_images)
                    nb_images_val += (cl_images - int((1 - FLAGS.val_split) * cl_images))
        return nb_images_tr, nb_images_val
    
    
    def ds_performance(self,ds, is_train_set, is_training):
        if FLAGS.cache_dataset:
            #TODO: it seems that once cached, the images will always be the same
            #https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache
            ds = ds.cache()
        if is_train_set and is_training:
            buffer_multiplier = 100 if FLAGS.image_size <= 32 else 5
            #TODO AWS: figure out sharding / different seed for workers
            ds = ds.shuffle(self.batch_size * buffer_multiplier, seed = self.seed,# + int(hvd_rank), 
                                        reshuffle_each_iteration = True)
            ds = ds.repeat(-1)
        ds = ds.batch(self.batch_size, drop_remainder = is_training)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds
     
        
    def get_preprocess_fn(self,is_training, is_pretrain):
        """Get function that accepts an image and returns a preprocessed image."""
        # Disable test cropping for small images (e.g. CIFAR)
        if FLAGS.image_size <= 32:
            test_crop = False
        else:
            test_crop = True
            
        if is_pretrain == False and FLAGS.use_old_data_aug:
            print("Using old data augmentation pipeline")
            return functools.partial(
                data_util.transform_crop,
                img_size=(FLAGS.image_size,FLAGS.image_size,3)
                )
        else:
            return functools.partial(
                    data_util.preprocess_image,
                    height=FLAGS.image_size,
                    width=FLAGS.image_size,
                    is_training=is_training,
                    color_distort=is_pretrain,
                    test_crop=test_crop,
                    cutout = FLAGS.aug_cutout)
    
    
    def single_group_ds(self,this_class):
        """   group classes listed in classes with sampling weights per class as one dataset.
        Args:
            this_class (dict}: single-element list of classes to group into 1
        Returns:
            dataset
        """
        # get integer weights -- [0.1, 0.2] becomes [1, 2], sample second class twice as first

        
        
        train_items = np.array([])
        train_im_idx = np.array([])
        val_items = np.array([])

        weights = {"AK":FLAGS.weight_AK, "BCC":FLAGS.weight_BCC, 
                   "BKL":FLAGS.weight_BKL, "DF":FLAGS.weight_DF,
                   "MEL":FLAGS.weight_MEL, "NV":FLAGS.weight_NV, 
                   "SCC":FLAGS.weight_SCC, "VASC":FLAGS.weight_VASC}
        
        ix_class = list(weights.keys()).index(this_class[0])
        ix_sep = ix_class * 10000 # biggest class has 8k images
        
        
        if FLAGS.oversample == True and self.phase == "pretrain":
            print(f"Oversampling minority class {this_class[0]} with weight {weights[this_class[0]]} for pretraining")
        elif FLAGS.oversample_f == True and self.phase == "finetune":
            print(f"Oversampling minority class {this_class[0]} with weight {weights[this_class[0]]} for finetuning")
        
        else:
            print(f"No Oversampling for {self.phase}ing for all classes")
            weights = {"AK":1, "BCC":1, 
                   "BKL":1, "DF":1,
                   "MEL":1, "NV":1, 
                   "SCC":1, "VASC":1}
        
        for _cl in this_class:
            files = np.array(tf.io.gfile.glob(os.path.join(self.root_folder, _cl, f'*.{FLAGS.img_ext}')))
            files = sorted(files)
            self.rng.shuffle(files)
            if len(files) == 0:
                raise ValueError(f"Class {_cl} had no items in its folder!")
            
            #no splitting train/val
            if not(self.create_val_split):
                train_items  = np.concatenate((train_items, np.repeat(files, weights[_cl])))
                img_ixes = np.char.mod('%d',(ix_sep + np.repeat(np.arange(len(files)), weights[_cl])))
                train_im_idx = np.concatenate((train_im_idx, img_ixes))

            #Splitting train and validation sets
            elif self.create_val_split:
                split_th = int((1 - FLAGS.val_split) * len(files))
                if len(files) - split_th < 1:
                    raise ValueError(f"Validation split {FLAGS.val_split} is too small for class {_cl}")
                train_items = np.concatenate((train_items, np.repeat(files[:split_th], weights[_cl])))
                img_ixes = np.char.mod('%d',(ix_sep + np.repeat(np.arange(split_th), weights[_cl])))
                train_im_idx = np.concatenate((train_im_idx, img_ixes))
                # we don't oversample the validation class
                val_items = np.concatenate((val_items, files[split_th:]))
                
        return train_items, val_items, train_im_idx
    
    
    
    def build_dataset(self,batch_size, dist, is_training, seed=None):
        if not seed:
            self.seed = int(time.time())
            print(f"No seed provided, generating one: seed = {self.seed} ")
        else:
            print(f"Using pre-defined seed = {seed}")
            self.seed = seed
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed=self.seed)
        
        
        train = None
        val = None
        preprocess_fn_pretrain = self.get_preprocess_fn(is_training, is_pretrain=True)
        preprocess_fn_finetune = self.get_preprocess_fn(is_training, is_pretrain=False)
        
        
        #Image augmentation / pre-processing map function
        def map_fn_train(row):
            """Load images and produces multiple transformations 
            of the same batch."""
            image_path = row[0]
            label_ix   = tf.strings.to_number(row[1], out_type=tf.dtypes.int32)
            img_ix = tf.strings.to_number(row[2], out_type=tf.dtypes.int32)

            raw = tf.io.read_file(image_path)
            image =  tf.image.decode_jpeg(raw)
            
            if is_training and self.phase == 'pretrain':
                xs = []
                for _ in range(2):  # Two transformations
                    xs.append(preprocess_fn_pretrain(image))
                image = tf.concat(xs, -1)
            else:
                image = preprocess_fn_finetune(image)
            label = tf.one_hot(label_ix, self.num_classes)
            
            if self.phase == 'pretrain':
                return image, label, img_ix
            else:
                return image, label
        
        def map_fn_val(row):
            """Load images and produces multiple transformations 
            of the same batch."""
            image_path = row[0]
            label_ix   = tf.strings.to_number(row[1], out_type=tf.dtypes.int32)
            raw = tf.io.read_file(image_path)
            image =  tf.image.decode_jpeg(raw)
            image = self.get_preprocess_fn(False, is_pretrain=False)(image)
            label = tf.one_hot(label_ix, self.num_classes)
            
            return image, label
        

        
        for idx, cl in enumerate(self.class_grouping):
            t, v, timx = self.single_group_ds(cl)
            if train is None:
                train = np.column_stack((t, np.repeat([idx], len(t)), timx))
                val = np.column_stack((v, np.repeat([idx], len(v))))
            else:
                train = np.vstack((train, np.column_stack((t, np.repeat([idx], len(t)), timx))))
                val = np.vstack((val, np.column_stack((v, np.repeat([idx], len(v))))))
        
        self.rng.shuffle(train, axis=0)
        
        #TODO AWS: include sharding to get a clean distributed training
        train_ds = tf.data.Dataset.from_tensor_slices(train)
        train_ds = train_ds.shard(dist.size(), dist.rank())
        train_ds = train_ds.map(map_fn_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = self.ds_performance(train_ds,is_train_set=True, is_training=is_training)
        
        if self.create_val_split:
            val_ds = tf.data.Dataset.from_tensor_slices(val)
            val_ds = val_ds.map(map_fn_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_ds = self.ds_performance(val_ds,is_train_set=False, is_training = False)
    
        else:
            val_ds = None
        
        return train_ds, val_ds

def main(argv):
    import matplotlib.pyplot as plt
    import numpy as np

    test = dataset_builder(FLAGS.data_dir,create_val_split = True, phase='pretrain')
    print(f"Number of classes: {test.num_classes}")
    print(f"Number of train examples: {test.num_train_examples}")
    print(f"Number of eval examples: {test.num_eval_examples}")
    
    tds,vds = test.build_dataset(32, is_training = True)
    
    iterator_tr = iter(tds)
    tr_images, tr_labels , tr_ix= next(iterator_tr)
    
    print(tr_ix.shape, "\n",tr_ix)
    
    iterator_vs = iter(vds)
    vs_images, vs_labels = next(iterator_vs)
    
    print('train minimum value = ',np.amin(tr_images))
    print('train maximum value = ',np.amax(tr_images))
    
    print('val minimum value = ',np.amin(vs_images))
    print('val maximum value = ',np.amax(vs_images))
    
    print('train first label =',tr_labels[0])
    print('val first label =',vs_labels[0])
    
    print(FLAGS.base_weights)
    
    # from collections import Counter
    # print('Train batch 0',Counter(np.argmax(tr_labels,-1)))
    # print('Val batch 0',Counter(np.argmax(vs_labels,-1)))
    # for i in range(100):
    #     tr_images, tr_labels = next(iterator_tr)
    #     assert np.amin(tr_images) < 10
    #     assert np.amax(tr_images) > 200
    #     print(f"Train batch {i+1}",Counter(np.argmax(tr_labels,-1)))
    
    
#    FOR VALIDATION images
    # plt.figure(figsize = (10,15))
    # for ix,(im,lab) in enumerate(zip(images[:20],labels[:20])): 
    #     plt.subplot(5, 4, ix+1)
    #     plt.imshow(im)
    #     img_cl = [["AK"],["BCC"],["BKL"],["DF"],["MEL"],["NV"],["SCC"],["VASC"]][np.argmax(lab)][0]
    #     plt.title(f"IMG {img_cl} train finet transf")
    # plt.savefig('../local_test/transf_images/ex_train_finetune_phase.png') 
    
    # FOR PRETRAINING images  
    plt.figure(figsize = (5,40))
    for ix,(im,lab,imix) in enumerate(zip(tr_images[:20],tr_labels[:20],tr_ix[:20])):         
        
        im1,im2 = np.split(im,2,-1)
        plt.subplot(10, 2, 2*ix+1)
        plt.imshow(im1/255)
        img_cl = [["AK"],["BCC"],["BKL"],["DF"],["MEL"],["NV"],["SCC"],["VASC"]][np.argmax(lab)][0]
        plt.title(f"{img_cl} No{imix} t1")
        plt.subplot(10, 2, 2*ix+2)
        plt.imshow(im2/255)
        plt.title(f"{img_cl} No{imix} t2")
    plt.savefig('../local_test/transf_images/ex_pretrain_with_imix.png')

    # FOR TRAINING finetune images  
    # plt.figure(figsize = (10,15))
    # for ix,(im,lab) in enumerate(zip(tr_images[:20],tr_labels[:20])): 
    #     plt.subplot(5, 4, ix+1)
    #     plt.imshow(im/255)
    #     img_cl = [["AK"],["BCC"],["BKL"],["DF"],["MEL"],["NV"],["SCC"],["VASC"]][np.argmax(lab)][0]
    #     plt.title(f"IMG {img_cl} train finet transf")
    # plt.savefig('../local_test/transf_images/ex_train_finetune_phase_wcutout.png') 
    
        
    
    
if __name__ == '__main__':
    app.run(main)
