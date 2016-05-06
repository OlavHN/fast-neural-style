from scipy import misc
import os
import time
import tensorflow as tf
import vgg
import model
import reader

tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 5e0, "Weight for content features loss")
tf.app.flags.DEFINE_integer("STYLE_WEIGHT", 1e2, "Weight for style features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 1e-5, "Weight for total variation loss")
tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat",
        "Path to vgg model weights")
tf.app.flags.DEFINE_string("MODEL_PATH", "models", "Path to read/write trained models")
tf.app.flags.DEFINE_string("TRAIN_IMAGES_PATH", "train2014", "Path to training images")
tf.app.flags.DEFINE_string("CONTENT_LAYERS", "relu4_2",
        "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("STYLE_LAYERS", "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1",
        "Which layers to extract style from")
tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")
tf.app.flags.DEFINE_string("STYLE_IMAGES", "style.png", "Styles to train")
tf.app.flags.DEFINE_float("STYLE_SCALE", 1.0, "Scale styles. Higher extracts smaller features")
tf.app.flags.DEFINE_string("CONTENT_IMAGES_PATH", None, "Path to content image(s)")
tf.app.flags.DEFINE_integer("IMAGE_SIZE", 256, "Size of output image")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 1, "Number of concurrent images to train on")

FLAGS = tf.app.flags.FLAGS

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.pack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.pack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

# TODO: Figure out grams and batch sizes! Doesn't make sense ..
def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_float(size / FLAGS.BATCH_SIZE)

    return grams

def get_style_features(style_paths, style_layers):
    with tf.Graph().as_default() as g:
        size = int(round(FLAGS.IMAGE_SIZE * FLAGS.STYLE_SCALE))
        images = tf.pack([reader.get_image(path, size) for path in style_paths])
        net, _ = vgg.net(FLAGS.VGG_PATH, images)
        features = []
        for layer in style_layers:
            features.append(gram(net[layer]))

        with tf.Session() as sess:
            return sess.run(features)

def main(argv=None):
    if FLAGS.CONTENT_IMAGES_PATH:
        content_images = reader.image(
                FLAGS.BATCH_SIZE,
                FLAGS.IMAGE_SIZE,
                FLAGS.CONTENT_IMAGES_PATH,
                epochs=1,
                shuffle=False,
                crop=False)
        generated_images = model.net(content_images / 255.)

        output_format = tf.saturate_cast(generated_images + reader.mean_pixel, tf.uint8)
        with tf.Session() as sess:
            file = tf.train.latest_checkpoint(FLAGS.MODEL_PATH)
            if not file:
                print('Could not find trained model in {}'.format(FLAGS.MODEL_PATH))
                return
            print('Using model from {}'.format(file))
            saver = tf.train.Saver()
            saver.restore(sess, file)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            i = 0
            start_time = time.time()
            try:
                while not coord.should_stop():
                    print(i)
                    images_t = sess.run(output_format)
                    elapsed = time.time() - start_time
                    start_time = time.time()
                    print('Time for one batch: {}'.format(elapsed))

                    for raw_image in images_t:
                        i += 1
                        misc.imsave('out{0:04d}.png'.format(i), raw_image)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)
        return

    if not os.path.exists(FLAGS.MODEL_PATH):
        os.makedirs(FLAGS.MODEL_PATH)

    style_paths = FLAGS.STYLE_IMAGES.split(',')
    style_layers = FLAGS.STYLE_LAYERS.split(',')
    content_layers = FLAGS.CONTENT_LAYERS.split(',')

    style_features_t = get_style_features(style_paths, style_layers)

    images = reader.image(FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE, FLAGS.TRAIN_IMAGES_PATH)
    generated = model.net(images / 255.)

    net, _ = vgg.net(FLAGS.VGG_PATH, tf.concat(0, [generated, images]))

    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(0, 2, net[layer])
        size = tf.size(generated_images)
        content_loss += tf.nn.l2_loss(generated_images - content_images) / tf.to_float(size)
    content_loss = content_loss / len(content_layers)

    style_loss = 0
    for style_gram, layer in zip(style_features_t, style_layers):
        generated_images, _ = tf.split(0, 2, net[layer])
        size = tf.size(generated_images)
        for style_image in style_gram:
            style_loss += tf.nn.l2_loss(tf.reduce_sum(gram(generated_images) - style_image, 0)) / tf.to_float(size)
    style_loss = style_loss / len(style_layers)

    loss = FLAGS.STYLE_WEIGHT * style_loss + FLAGS.CONTENT_WEIGHT * content_loss + FLAGS.TV_WEIGHT * total_variation_loss(generated)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)

    output_format = tf.saturate_cast(tf.concat(0, [generated, images]) + reader.mean_pixel, tf.uint8)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        file = tf.train.latest_checkpoint(FLAGS.MODEL_PATH)
        if file:
            print('Restoring model from {}'.format(file))
            saver.restore(sess, file)
        else:
            print('New model initilized')
            sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        try:
            while not coord.should_stop():
                _, loss_t, step = sess.run([train_op, loss, global_step])
                elapsed_time = time.time() - start_time
                start_time = time.time()
                if step % 100 == 0:
                    print(step, loss_t, elapsed_time)
                    output_t = sess.run(output_format)
                    for i, raw_image in enumerate(output_t):
                        misc.imsave('out{}.png'.format(i), raw_image)
                if step % 10000 == 0:
                    saver.save(sess, FLAGS.MODEL_PATH + '/fast-style-model', global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
