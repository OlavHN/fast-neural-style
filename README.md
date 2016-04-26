# Fast neural style transfer

In an attempt to learn Tensorflow I've implemented an Image Transformation Network as described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155) by Johnson et al.

This technique uses loss functions based on a perceptual similarity and style similarity as described by [Gatys et al](http://arxiv.org/abs/1508.06576) to train a transformation network to synthesize the style of one image with the content of arbitrary images. After it's trained for a particular style it can be used to generate stylized images in one forward pass through the transformer network as opposed to 500-2000 forward + backward passes through a pretrained image classification net which is the direct approach.

### Usage

To generate an image directly from style and content, typically to explore styles and parameters:

`python3 neural_style.py --CONTENT_IMAGE content.png --STYLE_IMAGES style.png`

Also see other settings and hyperparameters in neural_style.py

To train a model for fast stylizing:

`python3 fast-neural-style.py --TRAIN_IMAGES_PATH coco_img_path --STYLE_IMAGES style.png --BATCH_SIZE 4`

Where `--TRAIN_IMAGES_PATH` points to a directory of JPEGs to train the model. The paper uses the [COCO image dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) (13GB). With my 2GB GTX960 card I can do a batch_size of 3 images. The paper trains the model for 2 epochs (160.000/BATCH_SIZE iteration).

To generate images fast with an already trained model:

`python3 fast-neural-style.py --CONTENT_IMAGES path_to_images_to_transform`

### Requirements

- Python3.x
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [VGG-19 model](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)
- [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)

### TODO

- Automatic download for VGG-19 model and COCO dataset.
- Add a pretrained model
- Add example pictures / videos
- Resize when doing batch forward passes
- Test and fix for python2.7
- Add back tensorboard metrics

### Acknowledgement

- [Chainer implementation] (https://github.com/yusuketomoto/chainer-fast-neuralstyle)
- [Tensorflow Neural style implementation] (https://github.com/anishathalye/neural-style)
