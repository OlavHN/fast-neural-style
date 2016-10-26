from os import listdir
from os.path import isfile, join
import tensorflow as tf

mean_pixel = [123.68, 116.779, 103.939] # ImageNet average from VGG ..

def preprocess(image, size):
    shape = tf.shape(image)
    size_t = tf.constant(size, tf.float64)
    height = tf.cast(shape[0], tf.float64)
    width = tf.cast(shape[1], tf.float64)

    cond_op = tf.less(height, width)

    new_height, new_width = tf.cond(
        cond_op,
        lambda: (size_t, (width * size_t) / height),
        lambda: ((height * size_t) / width, size_t))

    resized_image = tf.image.resize_images(
            image,
            [tf.to_int32(new_height), tf.to_int32(new_width)],
            method=tf.image.ResizeMethod.BICUBIC)
    cropped = tf.image.resize_image_with_crop_or_pad(resized_image, size, size)

    return cropped

def get_image(path, size):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess(image, size)

def image(n, size, path, epochs=2, shuffle=True, crop=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if not shuffle:
        filenames = sorted(filenames)

    png = filenames[0].lower().endswith('png') # If first file is a png, assume they all are

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)

    processed_image = preprocess(image, size)
    return tf.train.batch([processed_image], n, dynamic_pad=True)
