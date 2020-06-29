# Visual Question Answering

## Table of Content
- [Introduction](#introduction)
- [Demo](#demo)
- [Model Overview](#model-overview)
- [Technical Aspect](#technical-aspect)
- [Running the app locally](#running-the-app-locally)
- [Project Directory Tree](#project-directory-tree)
- [Technologies Used](#technologies-used)
- [To Do](#to-do)
- [Contributions / Bug](#contributions--bug)
- [License](#license)

## Introduction
A simple Flask app to generate answer given an image and a natural language question about the image. The app uses a deep learning model, trained with Tensorflow, behind the scenes.

## Demo 
Link - https://youtu.be/pah91J4MnzI

[![](http://img.youtube.com/vi/pah91J4MnzI/0.jpg)](http://www.youtube.com/watch?v=pah91J4MnzI "Demo Video")

## Model Overview
Recent developments in Deep Learning has paved the way to accomplish tasks involving multimodal learning. Visual Question Answering (VQA) is one such challenge which requires high-level scene interpretation from images combined with language modelling of relevant Q&A. Given an image and a natural language question about the image, the task is to provide an accurate natural language answer. This is a Keras implementation of one such end-to-end system to accomplish the task.

The model architecture is based on the paper [Hierarchical Question-Image Co-Attention for Visual Question Answering](https://arxiv.org/pdf/1606.00061).

## Technical Aspect
The model used in the app is trained on [VQA 2.0](https://visualqa.org/download.html) dataset. The accuracy of the paper on this dataset is 54%. The model used in the Flask app has an accuracy of 49.20%.

## Running the app locally
*The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.*

First, clone this project to your local machine:
```
https://github.com/arya46/VQA_HieCoAtt.git

# change the working directory
cd VQA_HieCoAtt
```
Then install the required packages and libraries. Run the following command:
```
pip install -r requirements.txt
```
Everything is set now. Use the following command to launch the app:
```
python main.py
```
The app will run on `http://localhost:8080/` in the browser.

## Project Directory Tree
```
├── models 
│   ├── arch.py   #contains the model final model architecture
│   └── layers.py #contains the custom layers
├── pickles 
│   ├── complete_model.h5  #the trained Keras model
│   ├── labelencoder.pkl   #LabelEncoder object
│   └── text_tokenizer.pkl #Keras tokenizer object
├── templates 
│   ├── index.html
│   └── error.html 
├── utils 
│   ├── helper_functions.py
│   └── load_pickles.py
├── LICENSE
├── README.md
├── main.py
└── requirements.txt
```
## Technologies Used
- Programming Language: Python
- ML Tools/Libraries: Keras, Tensorflow, Scikit Learn, Numpy, Pandas
- Web Tools/Libraries: Flask, HTML

## To Do
- [ ] Deployement on Heroku

## Contributions / Bug
If you want to contribute to this project, or want to report a bug, kindly open an issue [here](https://github.com/arya46/VQA_HieCoAtt/issues/new).

## License
[LICENSE](https://github.com/arya46/VQA_HieCoAtt/blob/master/LICENSE)
