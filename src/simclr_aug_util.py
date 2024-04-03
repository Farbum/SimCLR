import tensorflow as tf
import tensorflow_addons as tfa
import functools
from absl import flags
import math
import numpy as np

FLAGS = flags.FLAGS

def random_apply(func, p, x):
    """Randomly apply function func to x with probability p."""
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32)), lambda: func(x), lambda: x)





######################  CROP AND RESIZE ###################
def _compute_crop_shape(image_height, image_width, aspect_ratio, crop_proportion):
    """Compute aspect ratio-preserving shape for central crop.
    The resulting shape retains `crop_proportion` along one side and a proportion
    less than or equal to `crop_proportion` along the other side.
    Args:
      aspect_ratio: Desired aspect ratio (width / height) of output.
      crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
      crop_height: Height of image after cropping.
      crop_width: Width of image after cropping.
    """
    image_width_float = tf.cast(image_width, tf.float32)
    image_height_float = tf.cast(image_height, tf.float32)
    
    def _requested_aspect_ratio_wider_than_image():
        crop_height = tf.cast(
            tf.math.rint(crop_proportion / aspect_ratio * image_width_float),
            tf.int32)
        crop_width = tf.cast(
            tf.math.rint(crop_proportion * image_width_float), tf.int32)
        return crop_height, crop_width
    
    def _image_wider_than_requested_aspect_ratio():
        crop_height = tf.cast(
            tf.math.rint(crop_proportion * image_height_float), tf.int32)
        crop_width = tf.cast(
            tf.math.rint(crop_proportion * aspect_ratio * image_height_float),
            tf.int32)
        return crop_height, crop_width
    
    return tf.cond(
        aspect_ratio > image_width_float / image_height_float,
        _requested_aspect_ratio_wider_than_image,
        _image_wider_than_requested_aspect_ratio)
  
  
def center_crop(image, height, width, crop_proportion):
    """Crops to center of image and rescales to desired size.
    Args:
      crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
      A `height` x `width` x channels Tensor holding a central crop of `image`.
    """
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    crop_height, crop_width = _compute_crop_shape(
        image_height, image_width, height / width, crop_proportion)
    offset_height = ((image_height - crop_height) + 1) // 2
    offset_width = ((image_width - crop_width) + 1) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width)
      
    image = tf.image.resize([image], [height, width],
                            method=tf.image.ResizeMethod.BICUBIC)[0]
    return image


def crop_and_resize(image, height, width):
    """Make a random crop and resize it to height `height` and width `width`.
    Args:
      image: Tensor representing the image.
      height: Desired image height.
      width: Desired image width.
    Returns:
      A `height` x `width` x channels Tensor holding a random crop of `image`.
    """
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = width / height
    image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        area_range=(0.08, 1.0),
        max_attempts=100,
        scope=None)
    return tf.image.resize([image], [height, width],
                           method=tf.image.ResizeMethod.BICUBIC)[0]




def random_crop_with_resize(image, height, width, p=1.0):
    """Randomly crop and resize an image.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      p: Probability of applying this transformation.
    Returns:
      A preprocessed image `Tensor`.
    """
    def _transform(image):  # pylint: disable=missing-docstring
        image = crop_and_resize(image, height, width)
        return image
    return random_apply(_transform, p=p, x=image)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
      image: `Tensor` of image data.
      bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
      scope: Optional `str` for name scope.
    Returns:
      (cropped image `Tensor`, distorted bbox `Tensor`).
    """
    with tf.name_scope(scope or 'distorted_bounding_box_crop'):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box
      
        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, target_height, target_width)
      
        return image


######################  COLOR DISTORTION ###################


def color_jitter(image, strength, random_order=True, impl='simclrv2'):
    """Distorts the color of the image.
    Args:
      image: The input image tensor.
      strength: the floating number for the strength of the color augmentation.
      random_order: A bool, specifying whether to randomize the jittering order.
      impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
          version of random brightness.
    Returns:
      The distorted image tensor.
    """
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    if random_order:
        return color_jitter_rand(
            image, brightness, contrast, saturation, hue, impl=impl)
    else:
        return color_jitter_nonrand(
            image, brightness, contrast, saturation, hue, impl=impl)


def to_grayscale(image, keep_channels=True):
    image = tf.image.rgb_to_grayscale(image)
    if keep_channels:
        image = tf.tile(image, [1, 1, 3])
    return image

def random_color_jitter(image, p=1.0, impl='simclrv2'):
    
    def _transform(image):
        color_jitter_t = functools.partial(
            color_jitter, strength=FLAGS.color_jitter_strength, impl=impl)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(to_grayscale, p=0.2, x=image)
    return random_apply(_transform, p=p, x=image)


def random_brightness(image, max_delta, impl='simclrv2'):
    """A multiplicative vs additive change of brightness."""
    if impl == 'simclrv2':
        factor = tf.random.uniform([], tf.maximum(1.0 - max_delta, 0),
                                   1.0 + max_delta)
        image = image * factor
    elif impl == 'simclrv1':
        image = tf.image.random_brightness(image, max_delta=max_delta)
    else:
        raise ValueError('Unknown impl {} for random brightness.'.format(impl))
    return image


def color_jitter_rand(image,
                      brightness=0,
                      contrast=0,
                      saturation=0,
                      hue=0,
                      impl='simclrv2'):
    """Distorts the color of the image (jittering order is random).
    Args:
      image: The input image tensor.
      brightness: A float, specifying the brightness for color jitter.
      contrast: A float, specifying the contrast for color jitter.
      saturation: A float, specifying the saturation for color jitter.
      hue: A float, specifying the hue for color jitter.
      impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
          version of random brightness.
    Returns:
      The distorted image tensor.
    """
    with tf.name_scope('distort_color'):
        def apply_transform(i, x):
            """Apply the i-th transformation."""
            def brightness_foo():
                if brightness == 0:
                    return x
                else:
                    return random_brightness(x, max_delta=brightness, impl=impl)
        
            def contrast_foo():
                if contrast == 0:
                    return x
                else:
                    return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
            def saturation_foo():
                if saturation == 0:
                    return x
                else:
                    return tf.image.random_saturation(
                      x, lower=1-saturation, upper=1+saturation)
            def hue_foo():
                if hue == 0:
                    return x
                else:
                    return tf.image.random_hue(x, max_delta=hue)
            x = tf.cond(tf.less(i, 2),
                    lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                    lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
            return x
    
        perm = tf.random.shuffle(tf.range(4))
        for i in range(4):
            image = apply_transform(perm[i], image)
            image = tf.clip_by_value(image, 0., 1.)
        return image




######################  GAUSSIAN BLUR ###################

def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
    """Blurs the given image with separable convolution.
    Args:
      image: Tensor of shape [height, width, channels] and dtype float to blur.
      kernel_size: Integer Tensor for the size of the blur kernel. This is should
        be an odd number. If it is an even number, the actual kernel size will be
        size + 1.
      sigma: Sigma value for gaussian operator.
      padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
    Returns:
      A Tensor representing the blurred image.
    """
    radius = tf.cast(kernel_size / 2, dtype=tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
    blur_filter = tf.exp(-tf.pow(x, 2.0) /
                         (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        # Tensorflow requires batched input to convolutions, which we can fake with
        # an extra dimension.
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred

def random_blur(image, height, width, p=1.0):
    """Randomly blur an image.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      p: probability of applying this transformation.
    Returns:
      A preprocessed image `Tensor`.
    """
    del width
    def _transform(image):
        sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
        return gaussian_blur(
            image, kernel_size=height//10, sigma=sigma, padding='SAME')
    return random_apply(_transform, p=p, x=image)


def random_cutout(image,h,w):
    image = tf.expand_dims(image, axis=0)
    half_side_cut = int(h/7)
    off_set1 = tf.random.uniform((), minval=half_side_cut, maxval= h - half_side_cut, dtype=tf.dtypes.int32)
    off_set2 = tf.random.uniform((), minval=half_side_cut, maxval= w - half_side_cut, dtype=tf.dtypes.int32)

    image = tfa.image.cutout(
        image, (2 * half_side_cut), offset=(off_set1,off_set2))
    return image

def batch_random_blur(images_list, height, width, blur_probability=0.5):
    """Apply efficient batch data transformations.
    Args:
      images_list: a list of image tensors.
      height: the height of image.
      width: the width of image.
      blur_probability: the probaility to apply the blur operator.
    Returns:
      Preprocessed feature list.
    """
    def generate_selector(p, bsz):
        shape = [bsz, 1, 1, 1]
        selector = tf.cast(
           tf.less(tf.random.uniform(shape, 0, 1, dtype=tf.float32), p),
           tf.float32)
        return selector

    new_images_list = []
    for images in images_list:
          images = images / 255.
          images_new = random_blur(images, height, width, p=1.)
          selector = generate_selector(blur_probability, tf.shape(images)[0])
          images = images_new * selector + images * (1 - selector)
          images = tf.clip_by_value(images, 0., 1.)
          images = images*255.
          new_images_list.append(images)
    
    return new_images_list





######################  preprocessing  ###################
def preprocess_for_train(image,
                         height,
                         width,
                         color_distort=True,
                         crop=True,
                         flip=True,
                         cutout=True,
                         impl='simclrv2'):
    """Preprocesses the given image for training.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      color_distort: Whether to apply the color distortion.
      crop: Whether to crop the image.
      flip: Whether or not to flip left and right of an image.
      impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
      version of random brightness.
    Returns:
      A preprocessed image `Tensor`.
    """
    if crop:
        image = random_crop_with_resize(image, height, width)
    if flip:
        image = tf.image.random_flip_left_right(image)
    if color_distort:
        image = random_color_jitter(image, impl=impl)
    if cutout:
        image = random_cutout(image,height,width)

    image = tf.reshape(image, [height, width, 3])   
    image = tf.clip_by_value(image, 0., 1.)
    image = image*255.
    return image


def preprocess_for_eval(image, height, width, crop=True):
    """Preprocesses the given image for evaluation.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      crop: Whether or not to (center) crop the test images.
    Returns:
      A preprocessed image `Tensor`.
    """
    if crop:
        image = center_crop(image, height, width, crop_proportion=FLAGS.val_center_crop_proportion)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    image = image*255.
    return image


def preprocess_image(image, height, width, is_training=False,
                     color_distort=True, test_crop=True, cutout=True):
    """Preprocesses the given image.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      is_training: `bool` for whether the preprocessing is for training.
      color_distort: whether to apply the color distortion.
      test_crop: whether or not to extract a central crop of the images
          (as for standard ImageNet evaluation) during the evaluation.
    Returns:
      A preprocessed image `Tensor` of range [0, 1].
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
        return preprocess_for_train(image, height, width, color_distort,cutout=cutout)
    else:
        return preprocess_for_eval(image, height, width, test_crop)

    
def transform_crop(image,img_size=(224,224,3)):
    """
    performs a series of random transformation on the input image
    Args:
        image: TF tensor to transform  
        aug_hparam (dict): dictionary of augmentation parameters for training image transformations

    Returns:
        transformed image
    """
    aug_hparam = {
        'aug_rot_prob' : 0.4019,
        'aug_zoom_prob' : 0.1925,
        'aug_flipv_prob' : 0.5,
        'aug_fliph_prob' : 0.5,
        'aug_cut_prob' : 0.3846,
        'aug_bright_prob' : 0.0232,
        'aug_contrast_prob' : 0.75,
        'aug_saturation_prob' : 0.452,
        'aug_bright_delta' : 0.3874,
        'aug_cont_lowfact' : 0.002,
        'aug_cont_upfact' : 2.575,
        'aug_rot_degree' : 45,
        'aug_rot_fill' : 'reflect',
        'aug_zoomin_fact' : 0.9,
        'aug_zoomout_fact' : 1.2,
        'aug_zoom_fill' : 'reflect',
        'aug_cutout_halfsize' : 32,
        'aug_saturation_fact_min' : 0.93,
        'aug_saturation_fact_max' : 2.26}
        
    rand_num = tf.random.uniform((),0,1)
    rs1 = tf.random.uniform((), minval=0, maxval=img_size[0], dtype=tf.dtypes.int32)
    rs2 = tf.random.uniform((), minval=0, maxval=img_size[1], dtype=tf.dtypes.int32)
    seed_s2 = (rs1,rs2)
    # Random brightness
    if rand_num <= aug_hparam["aug_bright_prob"]:  
        image = tf.image.stateless_random_brightness(image, 
                        max_delta = aug_hparam["aug_bright_delta"],seed=seed_s2)


    # Random saturation
    if rand_num <= aug_hparam["aug_saturation_prob"]:  
        minval = aug_hparam["aug_saturation_fact_min"]
        maxval = aug_hparam["aug_saturation_fact_max"]
        factor = tf.random.uniform((), minval=minval, maxval=maxval, dtype=tf.dtypes.float32)
        image = tf.image.adjust_saturation(image, factor) 

    # Random contrast
    if rand_num <= aug_hparam["aug_contrast_prob"]: 
        image = tf.image.stateless_random_contrast(image, 
                       lower = aug_hparam["aug_cont_lowfact"],
                       upper = aug_hparam["aug_cont_upfact"], seed = seed_s2)
    # flip vert
    if rand_num <= aug_hparam["aug_flipv_prob"]: 
        image = tf.image.flip_left_right(image)
    # flip hor
    if rand_num <= aug_hparam["aug_fliph_prob"]: 
        image = tf.image.flip_up_down(image)
    # Random rotation
    if rand_num <= aug_hparam["aug_rot_prob"]: 
        angle_rd = tf.random.uniform((),-aug_hparam["aug_rot_degree"]/360, aug_hparam["aug_rot_degree"]/360)
        image = tfa.image.rotate(image,2* math.pi*angle_rd,
                                 fill_mode=aug_hparam["aug_rot_fill"], interpolation='nearest')
    # Random Zoom
    if rand_num <= aug_hparam["aug_zoom_prob"]: 
        zoom_fact = tf.random.uniform((),aug_hparam["aug_zoomin_fact"],aug_hparam["aug_zoomout_fact"])

        new_size = [tf.cast(tf.floor(zoom_fact*tf.cast(tf.shape(image)[0],dtype = tf.dtypes.float32)),dtype = tf.dtypes.int32),
                    tf.cast(tf.floor(zoom_fact*tf.cast(tf.shape(image)[1],dtype = tf.dtypes.float32)),dtype = tf.dtypes.int32)]
        image = tf.cast(tf.image.resize(image,new_size,
                preserve_aspect_ratio=False,antialias=True),dtype = tf.dtypes.uint8)
    # Random crop
    image = tf.image.stateless_random_crop(
        image, size=[img_size[0], img_size[1], 3], seed=seed_s2)
    # Random Cutout
    if rand_num <= aug_hparam["aug_cut_prob"]:
        half_side_cut = aug_hparam["aug_cutout_halfsize"]
        off_set1 = tf.random.uniform((), minval=half_side_cut, maxval= img_size[0] - half_side_cut, dtype=tf.dtypes.int32)
        off_set2 = tf.random.uniform((), minval=half_side_cut, maxval= img_size[1] - half_side_cut, dtype=tf.dtypes.int32)

        image = tfa.image.cutout(
            tf.expand_dims(image, axis=0), (2 * half_side_cut), offset=(off_set1,off_set2))
        image = tf.reshape(image, image.shape[1:])
    return image
    

def get_exh_offset(img_shape,val_ix_crop):  
    """
    compute the validation crop offset based on crop index and number of crops
    Args:
        img_shape: image shape, TF tensor  
    Returns:
        x and y offsets corresponding to the crop index
    """
    img_size_tf = tf.constant(img_size)
    n_crop_side = int(np.sqrt(FLAGS.val_nb_crops))
    delta_x = tf.floor(tf.abs(img_shape[0] - img_size_tf[0]) / (n_crop_side - 1))
    delta_y = tf.floor(tf.abs(img_shape[1] - img_size_tf[1]) / (n_crop_side - 1))
    x_off_set  = delta_x * (val_ix_crop // n_crop_side)
    y_off_set  = delta_y * (val_ix_crop % n_crop_side)
    x_off_set_cast = tf.cast(x_off_set,dtype=tf.int32)
    y_off_set_cast = tf.cast(y_off_set,dtype=tf.int32)
    return x_off_set_cast, y_off_set_cast
