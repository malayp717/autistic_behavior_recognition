# Autistic Behavior Recognition

This repository implements deep learning-based models for recognizing and summarizing behaviors in videos. The models focus on classifying normal behaviors and distinguishing between various maladaptive behaviors such as tantrums, self-injury, and hand flapping. The work integrates vision-only models (e.g., Video Swin Transformer) and language-assisted models (e.g., VST with CLIP) for robust behavior classification.

## Problem Description

The problem description can be found in the file "Assignment on Video Classification and Summarization - Research Engineer.pdf"

## Approach

There are several steps which we need to perform to run this pipeline. 

### 1. Download Dataset

The script `download.py` downloads mp4 videos from youtube and google drive after reading it from the `ASBD.csv` file.
The parent directory of the downloaded videos will be the `data_dir` specified in the `config.yaml`.

- The videos containing abnormal child behavior will be downloaded at the path: `{data_dir}/abnormal/{abnormality_category}`
- The videos containing normal child behavior will be downloaded at the path: `{data_dir}/normal/`

Run this command:

`python download.py`

### 2. Preprocess Dataset

The downloaded mp4 files need to be preprocessed to obtain cleaner data that include target children performing
ASD behaviors only. To do this we use YOLOv5 model to detect regions containing children and then crop those regions.
We segment video to prepare clips of 30 frames each.

To preprocess the dataset run this command:

`python preproc.py`

### 3. Train and evaluate the model

We have used 3 different model architectures:

#### a) Video Swin Transformer (VST) Model Architecture

The VST implementation has been taken from [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer). We freeze all the layers except the last 
classification head. We modify the classification head to output a feature vector of `num_classes` dimensionality.
This has been implemented in the class `VST` in the file `VST.py`.

#### b) Language-Assisted Video Swin Transformer

We extend VST by integrating text descriptions using the CLIP text encoder as discussed in [this paper](https://arxiv.org/pdf/2211.09310). 
This has been implemented in the class `VSTWithCLIP` in the file `VST.py`.
To train a model using this architecture we implemented a custom loss function which introduces a contrastive loss between 
video features and text features. This loss function has been implemented in the class `VSTWithClipLoss` in the file `loss.py`.

#### c) Language-Assisted Video Swin Transformer with detailed class descriptions

The model architecture is same as above. The class descriptions are detailed instead of being one word.

To train the model and obtain model checkpoints run:

`python train.py`

### 4. Experimental Settings

We followed the same experimental settings as described in the section 4.1 of [this paper](https://arxiv.org/pdf/2211.09310).

#### Note:
This repository will not run as it is since we have not provided the raw dataset due to privacy concerns.