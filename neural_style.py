import time
import tensorflow as tf
import vgg
import reader

tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 5e0, "Weight for content features loss")
tf.app.flags.DEFINE_integer("STYLE_WEIGHT", 1e2, "Weight for style features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 1e-5, "Weight for total variation loss")
tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat",
        "Path to vgg model weights")
tf.app.flags.DEFINE_string("CONTENT_LAYERS", "relu4_2",
        "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("STYLE_LAYERS", "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1",
        "Which layers to extract style from")
tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")
tf.app.flags.DEFINE_string("STYLE_IMAGES", "style.png", "Styles to train")
tf.app.flags.DEFINE_float("STYLE_SCALE", 1.0, "Scale styles. Higher extracts smaller features")
tf.app.flags.DEFINE_float("LEARNING_RATE", 10., "Learning rate")
tf.app.flags.DEFINE_string("CONTENT_IMAGE", "content.jpg", "Content image to use")
tf.app.flags.DEFINE_boolean("RANDOM_INIT", True, "Start from random noise")
tf.app.flags.DEFINE_integer("NUM_ITERATIONS", 1000, "Number of iterations")
tf.app.flags.DEFINE_integer("IMAGE_SIZE", 256, "Size of output image")

FLAGS = tf.app.flags.FLAGS

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.pack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.pack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

# TODO: Okay to flatten all style images into one gram?
def gram(layer):
    shape = tf.shape(layer)
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.pack([-1, num_filters]))
    gram = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(size)

    return gram

# TODO: Different style scales per image.
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

def get_content_features(content_path, content_layers):
    with tf.Graph().as_default() as g:
        image = tf.expand_dims(reader.get_image(content_path, FLAGS.IMAGE_SIZE), 0)
        net, _ = vgg.net(FLAGS.VGG_PATH, image)
        layers = []
        for layer in content_layers:
            layers.append(net[layer])

        with tf.Session() as sess:
            return sess.run(layers + [image])

def main(argv=None):
    style_paths = FLAGS.STYLE_IMAGES.split(',')
    style_layers = FLAGS.STYLE_LAYERS.split(',')
    content_path = FLAGS.CONTENT_IMAGE
    content_layers = FLAGS.CONTENT_LAYERS.split(',')

    style_features_t = get_style_features(style_paths, style_layers)
    *content_features_t, image_t = get_content_features(content_path, content_layers)

    image = tf.constant(image_t)
    random = tf.random_normal(image_t.shape)
    initial = tf.Variable(random if FLAGS.RANDOM_INIT else image)

    net, _ = vgg.net(FLAGS.VGG_PATH, initial)

    content_loss = 0
    for content_features, layer in zip(content_features_t, content_layers):
        layer_size = tf.size(content_features)
        content_loss += tf.nn.l2_loss(net[layer] - content_features) / tf.to_float(layer_size)
    content_loss = FLAGS.CONTENT_WEIGHT * content_loss / len(content_layers)

    style_loss = 0
    for style_gram, layer in zip(style_features_t, style_layers):
        layer_size = tf.size(style_gram)
        style_loss += tf.nn.l2_loss(gram(net[layer]) - style_gram) / tf.to_float(layer_size)
    style_loss = FLAGS.STYLE_WEIGHT * style_loss / (len(style_layers) * len(style_paths))

    tv_loss = FLAGS.TV_WEIGHT * total_variation_loss(initial)

    total_loss = content_loss + style_loss + tv_loss

    train_op = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE).minimize(total_loss)

    output_image = tf.image.encode_png(tf.saturate_cast(tf.squeeze(initial) + reader.mean_pixel, tf.uint8))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        start_time = time.time()
        for step in range(FLAGS.NUM_ITERATIONS):
            _, loss_t = sess.run([train_op, total_loss])
            elapsed = time.time() - start_time
            start_time = time.time()
            print(step, elapsed, loss_t)
        image_t = sess.run(output_image)
        with open('out.png', 'wb') as f:
            f.write(image_t)

if __name__ == '__main__':
    tf.app.run()
