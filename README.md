# Fast neural style transfer

Based on "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" by Johnson et al.

Direct neural style as described in Gatys et al.
`python3 neural_style.py --CONTENT_IMAGE content.png --STYLE_IMAGES style.png`

Train a model for fast neural style
`python3 fast-neural-style.py --TRAIN_IMAGES_PATH coco_img_path --BATCH_SIZE 3 --MODEL_PATH dir_to_save_model`

And the run fast queries!
`python3 fast-neural-style.py --MODEL_PATH dir_to_read_model --CONTENT_IMAGES path_to_images_to_transform`

TODO:

- Add download for VGG-19 model (checksum 8ee3263992981a1d26e73b3ca028a123)
- Add download for COCO image dataset
- Add a pretrained model
- Add more readme and examples
- Figure out the dots TV doesn't seem to remove ..
