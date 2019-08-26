# YeastNet: Deep Learning Semantic Segmentation for Budding Yeast Single Cell Analysis

YeastNet is a library for analysing live-cell fluorescence microscopy of budding yeast cells,  written in Python3 using PyTorch 1.0.

## Requirements

YeastNet works best using a conda environment and has been tested on a win-64 system and a x64 Ubuntu system. 


## Getting Started

1) Open a conda shell and use the environment.yml file to build a new conda environment.

Linux: ```conda env create -f environment.yml```

PC: ```conda env create -f environmentPC.yml```

2) Activate the new conda environment

```conda activate yeast-net```

3) Navigate to the folder containing the BrightField Images and run the Tracking Pipeline

```python .\trackingPipeline.py -f 'Z2' -p True```

## Details

Options for the trackingPipeline are available using the command:

```python trackingPipeline --help```

Bright Field Images by default should be in the path "./Images/BF/."
GFP Images should by default should be in the path "./Images/GFP/."

This library will use CUDA-based GPU computation for neural net inference by default. If no supported GPU is available, inference will complete using the CPU.  


## Files do Download

You can download the Images with Ground Truth Masks at:
<https://drive.google.com/open?id=17fEUZdODqfi0GwK9OQi3W3oKPX3vJHua>

You can download a set of YeastNet2 parameters at:
<https://drive.google.com/open?id=10nGfO99JcNNtZXyWDtaQN_HmYaPQ6ga9>

Export and place the YeastNet2 Model and the Images folder into the main directory.