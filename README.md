# High SNR was extracted from the dataset by Crad-CAM

PyTorch implementation of Grad-CAM (Gradient-weighted Class Activation Mapping) 
## Requirements

Python 3.+

```
$ pip install click opencv-python matplotlib tqdm numpy
$ pip install "torch>=0.4.1" torchvision
```

## Basic usage
To run the algorithm, use the following command:
```
python main.py [DEMO_ID] [OPTIONS]
```

Options:

* ```--image-paths```: image path, which can be provided multiple times (required)
* ```--arch```: a model name from ```torchvision.models```, e.g. "resnet152" (required)
* ```--target-layer```: a module name to be visualized, e.g. "layer4.2" (required)
* ```--topk```: the number of classes to generate (default: 3)
* ```--output-dir```: a directory to store results (default: ./results)
* ```--cuda/--cpu```: GPU or CPU

## Algorithm Details
The algorithm performs the following steps:

* Load the input image.
* Apply Grad-CAM to obtain the class activation map (CAM) for the specified target layer.
* Convert the CAM to a color map using the Blue-White-Red (bwr) colormap.
* Convert the color map from BGR to HSV color space.
* Threshold the HSV image to extract regions with high SNR.
* Save the thresholded image as the output.

The resulting image contains the high SNR feature extracted from the input image.
<img src="https://github.com/AIcode0608/H-SNR/blob/main/assets/H-SNR.png" width="660px">

## Example

To generate visualization maps for multiple images using a ResNet152 model and the "layer4" target layer, run the following command:
```bash
python main.py demo1 -a resnet152 -t layer4   -i samples/1.jpg -i samples/2.jpg # You can add more images
```
The high SNR feature extraction results will be stored in the './results' directory.
