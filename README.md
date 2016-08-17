# Fast neural style transfer

A short writeup and example images are up on [my blog](http://olavnymoen.com/2016/07/07/image-transformation-network).

In an attempt to learn Tensorflow I've implemented an Image Transformation Network as described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155) by Johnson et al.

This technique uses loss functions based on a perceptual similarity and style similarity as described by [Gatys et al](http://arxiv.org/abs/1508.06576) to train a transformation network to synthesize the style of one image with the content of arbitrary images. After it's trained for a particular style it can be used to generate stylized images in one forward pass through the transformer network as opposed to 500-2000 forward + backward passes through a pretrained image classification net which is the direct approach.

### Update Oct 23rd
- Changed upscale method to the one mentioned in http://distill.pub/2016/deconv-checkerboard/
- Better padding befoer each convolution to avoid border effects
- For some reason, padding the original image before the transformation image removes a lot of noise.
- "Instance normalization" instead of batch normalization as mentioned in [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
- No longer python3.

While the results are now much better, I'm still not sure why the original implementation didn't perform as well as Johnsons original work (Now published [here](https://github.com/jcjohnson/fast-neural-style))

### Usage

First get the dependecies (COCO training set images and VGG model weights):

`./get_files.sh`

To generate an image directly from style and content, typically to explore styles and parameters:

`python neural_style.py --CONTENT_IMAGE content.png --STYLE_IMAGES style.png`

Also see other settings and hyperparameters in neural_style.py

To train a model for fast stylizing first download dependences (training images and VGG model weights):

`./get_files.sh`

Then start training:

`python fast-neural-style.py --STYLE_IMAGES style.png --NAME=my_model`

Where `--TRAIN_IMAGES_PATH` points to a directory of JPEGs to train the model. `--NAME` is used for tensorboard statistics and file name of model weights. The paper uses the [COCO image dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) (13GB).

To generate images fast with an already trained model:

`python inference.py --IMAGE_PATH=my_content.jpg --NAME=my_model`

### Requirements

- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [VGG-19 model](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
- [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)

### TODO

- Add a pretrained model
- Add example pictures / videos

### Acknowledgement

- [Chainer implementation] (https://github.com/yusuketomoto/chainer-fast-neuralstyle)
- [Tensorflow Neural style implementation] (https://github.com/anishathalye/neural-style) (Both inspiration and copied the VGG code)
