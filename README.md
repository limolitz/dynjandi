# Dynjandi Webcam Processing Tool

This tool processes webcam input to allow for effects, mainly background masking so far.

## Features

* Simple background masking by comparing the current image with a base image
* Replace background with a green screen for further processing

## How-To (Ubuntu)

* Install `v4l2loopback-dkms`.
* Activate virtual webcam `sudo modprobe v4l2loopback video_nr=3 card_label="Dynjandi Output"`
* Install python3 venv, and install requirements
* Get base image `./get_base.py` (try to remove yourself from the image)
* Run main script `./get_diff.py`
* Profit! You can further process the image, e.g. with OBS Studio and make yourself float in space!

## Possible future features

* Better masking
** a machine learning approach seems to work better, but I'm not touching NodeJS with a ten-foot pole [1](https://elder.dev/posts/open-source-virtual-background/)

## Naming

Dynjandi is named after the Icelandic waterfall of the same name.

## License

The MIT License, copyright (c) 2020 wasmitnetzen