import uuid
import os
import time
import tensorflow as tf
import vgg
import model
import reader

tf.app.flags.DEFINE_string("NAME", "", "Name of this run")
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
tf.app.flags.DEFINE_integer("IMAGE_SIZE", 256, "Size of output image")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 4, "Number of concurrent images to train on")

FLAGS = tf.app.flags.FLAGS

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.pack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.pack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_float(width * height * num_filters)

    return grams

def get_style_features(style_paths, style_layers):
    with tf.Graph().as_default() as g:
        size = int(round(FLAGS.IMAGE_SIZE * FLAGS.STYLE_SCALE))
        images = tf.pack([reader.get_image(path, size) for path in style_paths])

        net, _ = vgg.net(FLAGS.VGG_PATH, images - reader.mean_pixel)
        features = []
        for layer in style_layers:
            features.append(gram(net[layer]))

        with tf.Session() as sess:
            return sess.run(features)

def main(argv=None):
    run_id = FLAGS.NAME if FLAGS.NAME else str(uuid.uuid4())
    model_path = '%s/%s' % (FLAGS.MODEL_PATH, run_id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    summary_path = '%s/%s' % (FLAGS.SUMMARY_PATH, run_id)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    style_paths = FLAGS.STYLE_IMAGES.split(',')
    style_layers = FLAGS.STYLE_LAYERS.split(',')
    content_layers = FLAGS.CONTENT_LAYERS.split(',')

    style_features_t = get_style_features(style_paths, style_layers)

    images = reader.image(FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE, FLAGS.TRAIN_IMAGES_PATH)
    generated = model.net(images - reader.mean_pixel, training=True)

    # Put both generated and training images in same batch through VGG net for efficiency
    net, _ = vgg.net(FLAGS.VGG_PATH, tf.concat(0, [generated, images]) - reader.mean_pixel)

    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(0, 2, net[layer])
        size = tf.size(generated_images)
        shape = tf.shape(generated_images)
        width = shape[1]
        height = shape[2]
        num_filters = shape[3]
        content_loss += tf.nn.l2_loss(generated_images - content_images) / tf.to_float(size)
    content_loss = content_loss

    style_loss = 0
    for style_grams, layer in zip(style_features_t, style_layers):
        generated_images, _ = tf.split(0, 2, net[layer])
        size = tf.size(generated_images)
        for style_gram in style_grams:
            style_loss += tf.nn.l2_loss(gram(generated_images) - style_gram) / tf.to_float(size)
    style_loss = style_loss / len(style_paths)

    tv_loss = total_variation_loss(generated)

    loss = FLAGS.STYLE_WEIGHT * style_loss + FLAGS.CONTENT_WEIGHT * content_loss + FLAGS.TV_WEIGHT * tv_loss

    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)

    # Statistics
    with tf.name_scope('losses'):
        tf.scalar_summary('content loss', content_loss)
        tf.scalar_summary('style loss', style_loss)
        tf.scalar_summary('regularizer loss', tv_loss)
    with tf.name_scope('weighted_losses'):
        tf.scalar_summary('weighted content loss', content_loss * FLAGS.CONTENT_WEIGHT)
        tf.scalar_summary('weighted style loss', style_loss * FLAGS.STYLE_WEIGHT)
        tf.scalar_summary('weighted regularizer loss', tv_loss * FLAGS.TV_WEIGHT)
        tf.scalar_summary('total loss', loss)
    tf.image_summary('original', images)
    tf.image_summary('generated', generated)

    summary = tf.merge_all_summaries()

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter(summary_path, sess.graph)

        saver = tf.train.Saver(tf.all_variables())
        file = tf.train.latest_checkpoint(model_path)
        sess.run([tf.initialize_all_variables(), tf.initialize_local_variables()])
        if file:
            print('Restoring model from {}'.format(file))
            saver.restore(sess, file)

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
                    summary_str = sess.run(summary)
                    writer.add_summary(summary_str, step)
                if step % 10000 == 0:
                    saver.save(sess, model_path + '/fast-style-model', global_step=step)
        except tf.errors.OutOfRangeError:
            saver.save(sess, model_path + '/fast-style-model-done')
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
