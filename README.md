# SAS-AIED2020
The repository for the AIED Conference 2020

## Prerequisites
Install the following dependencies:
 - Transformers (by Hugginface)
 - PyTorch

For training:
 - Tensorflow
 - Tensorboard
 - Sklearn
 - Apex (fp16)

For inference:
 - Flask
 - Flask CORS

## Setup models
The trained models are available in [Releases](https://github.com/28Smiles/SAS-AIED2020/releases) just download and extract the model in its corresponding (same name) folder.

## Interact with the model
We provide a html file called `frontend.html`, simply issue the command in the folder of one of the models contained in the `log` folder:
```
  flask run
```
And open the `frontend.html` in your browser.
