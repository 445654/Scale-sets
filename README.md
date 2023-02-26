# Scale Sets

This is an implementation based on the paper
[Scale-Sets Image Analysis](https://www.hds.utc.fr/~cocquere/dokuwiki/_media/fr/scale-sets_ijcv06.pdf)
by LAURENT GUIGUES, JEAN PIERRE COCQUEREZ and HERVÉ LE MEN,  
which talks about the segmentation of an image based on the scale-sets representation.

## Requirements

- [opencv](https://docs.opencv.org/4.7.0/d7/d9f/tutorial_linux_install.html)
- [argparse](https://github.com/p-ranav/argparse#building-installing-and-testing)

## Usage

```
Usage: scale set [--help] [--version] image lambda

Positional arguments:
  image         path to input image 
  lambda        varies from 0 to 1, with 0 meaning no merging and 1 meaning all merging 

Optional arguments:
  -h, --help    shows help message and exits 
  -v, --version prints version information and exits 
```

```bash
./scale-sets example.jpg 0.5
```

TODO

## Algorithm

The scale-sets representation consists of segmenting the image into a set of regions,
which is a combination of smaller regions that depend on a lambda factor input by the user.  
The lambda factor is a scale indicating the rate of regions that will be combined into a larger region.  
A lambda factor of 1 will result in the original image (with every pixel being a region),
while a lambda factor of 0 will result in a single region.

The combination of two regions is determined by comparing their energies.  
A region will merge with another region if the resulting energy is lower than the one of any other possible
combination.  
Here, the energy of a region is its variance, plus its perimeter multiplied by the lambda factor.

The algorithm starts by creating a region for each pixel in the image.  
Then, it iterates over all the regions to merge them with the constraint describe above, for the desired lambda.