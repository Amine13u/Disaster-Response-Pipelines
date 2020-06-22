# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)
5. [Instructions](#Instructions)

## Installation 

Beyond the Anaconda distribution of Python, the following packages need to be installed for nltk :
*   punkt
*   wordnet
*   averaged_perceptron_tagger

## Project Motivation

The objective of this project is to apply data engineering, natural language processing, and machine learning skills to analyze message reported by people during disasters to build a model for an API that classifies disaster messages.
These messages could potentially be sent to appropriate disaster relief agencies.
What makes this project great is that it helps people and potentially save lives.

## File Descriptions 

There are three main folders :
1. data :
   * `disaster_categories.csv` : dataset including all the categories
   * `disaster_messages.csv` : dataset including all the messages
   * `process_data.py` : ETL pipeline script to read, clean, and save data into a database
   * `DisasterResponse.db` : output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models :
   * `train_classifier.py` : machine learning pipeline script to train and export a classifier
   * `classifier.pkl` : output of the machine learning pipeline, i.e. a trained classifier
3. app : 
   * `run.py` : Flask file to run the web application
   * `templates` : contains html file for the web application

## Licensing, Authors, Acknowledgements

Credits must be given to [Udacity](https://www.udacity.com/) for the starter codes and [Figure Eight](https://appen.com/) for providing the data used in this project.
Feel free to use the code here as you would like ! 

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv   data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db  models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
        `python run.py`

3. Go to http://0.0.0.0:3001/