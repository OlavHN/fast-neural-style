#!/bin/sh

if [ ! -f "imagenet-vgg-verydeep-19.mat" ]; then
    echo "Downloading VGG imagenet weights"
    curl http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat > imagenet-vgg-verydeep-19.mat
else
    echo "VGG imagenet weights already found"
fi
if [ ! -d "train2014" ]; then
    echo "Downloading COCO image dataset"
    curl http://msvocds.blob.core.windows.net/coco2014/train2014.zip > train2014.zip
    unzip train2014.zip
    rm train2014.zip
else
    echo "Already found COCO image dataset"
fi
