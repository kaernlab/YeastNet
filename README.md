# YeastNet: Deep Learning Semantic Segmentation for Budding Yeast Single Cell Analysis

YeastNet is a library for analysing live-cell fluorescence microscopy of budding yeast cells,  written in Python3 using PyTorch 0.4.

## Requirements

YeastNet works best using a conda environment and has not been tested on anything but a win-64 system. 


## Getting Started

1) Open a conda shell and use the environment.yml file to build a new conda environment.

```conda env create -f environment.yml```

2) Activate the new conda environment

```conda activate yeastnet-env```

3) Navigate to the folder containing the BrightField Images and run the Tracking Pipeline

```python trackingPipeline```

## Details

Options for the trackingPipeline are available using the command:

```python trackingPipeline --help```

This library will use CUDA-based GPU computation for neural net inference by default. If no supported GPU is available, inference will complete using the CPU.  