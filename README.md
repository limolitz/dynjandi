# Dynjandi Webcam Processing Tool

This tool processes webcam input to allow for effects, mainly background masking so far.

## Features

* CLI to toggle features, show FPS and processing times
	* Method of background identification
    * Simple background masking by comparing the current image with a base image
    * Tensorflow based approach
  * Background replacement
    * Replace background with a green screen for further processing
    * Replace background with a blurred version of the background

## How-To (Ubuntu)

* Install `v4l2loopback-dkms`.
* Activate virtual webcam `sudo modprobe v4l2loopback video_nr=3 card_label="Dynjandi Output"`
* Install dependencies, python3 venv, and requirements
* Get base image `./get_base.py` (try to remove yourself from the image)
* Download trained model from [here](https://github.com/anilsathyan7/Portrait-Segmentation)
* Run main script `./get_diff.py`
* Profit! You can further process the image, e.g. with OBS Studio and make yourself float in space!

## Possible future features

* Better masking
  * A machine learning approach seems to work better ([[1]](https://elder.dev/posts/open-source-virtual-background/) [[2]](https://news.ycombinator.com/item?id=22823070)), but I'm not touching NodeJS with a ten-foot pole
* Improve speed, I currently get about 15 FPS at 1280x720

## Naming

Dynjandi is named after the Icelandic waterfall of the same name.

## License

The MIT License, copyright (c) 2020 wasmitnetzen
