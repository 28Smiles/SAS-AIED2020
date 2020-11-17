# SAS-AIED2020
The repository for the paper [Investigating Transformers for Automatic Short Answer Grading](https://www.springerprofessional.de/investigating-transformers-for-automatic-short-answer-grading/18148200) at the AIED Conference 2020. The Code quality is really bad and does not use the full power of the new transformers api. If creating something simmilar please use the Hugginface Trainer. 

The Code allows you to train a model with the tested hyperparameters on the full dataset.

## Prerequisites
Install the following dependencies:
 - Transformers (by Hugginface)
 - PyTorch

Dataset:

[Download from here](https://www.cs.york.ac.uk/semeval-2013/task7/index.php%3Fid=main-task-guidelines.html)

For training:
 - Tensorflow
 - Tensorboard
 - Sklearn
 - Apex (fp16)

For inference:
 - Flask
 - Flask CORS

## Interact with the model
We provide a html file called `frontend.html`, simply issue the command in the folder of one of the models contained in the `log` folder:
```
  flask run
```
And open the `frontend.html` in your browser.
