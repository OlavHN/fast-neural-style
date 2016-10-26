import reader
import tensorflow as tf
import model

tf.app.flags.DEFINE_string("MODEL_PATH", "models", "Path to read/write trained models")
tf.app.flags.DEFINE_string("NAME", "", "For finding the model")
tf.app.flags.DEFINE_string("IMAGE_PATH", "", "Path to image to trainsform")
FLAGS = tf.app.flags.FLAGS

model_path = FLAGS.MODEL_PATH + '/' + FLAGS.NAME

def main(argv=None):
    with open(FLAGS.IMAGE_PATH, 'rb') as f:
        jpg = f.read()

    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(jpg, channels=3), tf.float32) * 255.
    images = tf.expand_dims(image, 0)

    generated_images = model.net(images - reader.mean_pixel, training=False)

    output_format = tf.cast(generated_images, tf.uint8)
    jpegs = tf.map_fn(lambda image: tf.image.encode_jpeg(image), output_format, dtype=tf.string)

    with tf.Session() as sess:
        file = tf.train.latest_checkpoint(model_path)
        if not file:
            print('Could not find trained model in %s' % model_path)
            return
        print('Using model from %s' % file)
        saver = tf.train.Saver()
        saver.restore(sess, file)

        images_t = sess.run(jpegs)
        with open('res.jpg', 'wb') as f:
            f.write(images_t[0])

if __name__ == '__main__':
    tf.app.run()
