# Facial Verification [PyTorch]

[demo.webm](https://github.com/user-attachments/assets/913991d9-8a42-492f-8781-279d73616d66)

## Introduction
This application offers a convenient and secure login solution using facial recognition technology. Users can easily access their accounts by simply presenting their face to the camera, without the need for traditional login credentials.
## Key Features
* Robust facial verification model built using PyTorch
* Seamless user authentication experience
* Secure data handling and storage
* Customizable model training with flexible parameters
* Support for both local and cloud-based model training
## Model training
Users have several options to train the facial verification model:
* Download [lfw](http://vis-www.cs.umass.edu/lfw/#download) dataset
* Install the required dependencies by running `pip install -r requirements.txt`
#### Local Training:
* Run `python3 train_model.py -sp path/to/input/folder` to train the model with default parameters on a local dataset.
* Use the -nl flag if you have already downloaded the dataset.
#### Cloud-based Training:
* Run all cells in the train_azure.ipynb notebook in the train_azure folder to train the model on Azure with default parameters.
#### Custom Training:
* Run `python3 train_model.py -b batch_size -lr learning_rate` to train the model with your preferred batch size and learning rate.

## Web Application
Users can access the web application by running the Website branch.
#### Front-End
The front-end of the application is located in the user-interface folder. To set up and run the front-end, follow these steps:
* Navigate to the user-interface folder.
* Install the required dependencies by running npm install.
* Start the front-end development server by running npm start.
#### Back-End
The back-end of the application is located in the server-side folder. To set up and run the back-end, follow these steps:
* Navigate to the server-side folder.
* Install the required dependencies by running pip install -r requirements.txt.
* Set up an Azure SQL database to store the features of faces.
* Start the back-end server by running python3 app.py.
