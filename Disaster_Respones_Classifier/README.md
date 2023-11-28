### Project Motivation  

During a disaster, millions and millions of messages are often broadcasted either directly or via social media. We will probably see 1 in 1000 messages that are relevant. Few important words like water, blocked road, medical supplies are used during a disaster response. We have a categorical dataset with which we can train an ML model to see if we identify which messages are relevant to disaster response.

In this project three main features of a data science project have been utilized:

1. **Data Engineering** - The Extract, Transform and Load procedure are firstly conducted, then the resulted data is prepared for model training. For preparation, the data are cleaned by removing abnormalities (**ETL pipeline**) then the `nltk` is used to tokenize and lemmatize the data (**NLP Pipeline**).
2. **Model Training** - [ADABoost Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) is utilized to create the **ML pipeline**.
3. **Model Deployment** - `fastAPI` and `Jinja2` are used to construct the API and serve HTML response.  

**Interface**
![Main web page](https://github.com/stevenpham1996/Data-Science/blob/8d9f5adbd98149f99f112c954cdf66f04ef99d36/Disaster_Respones_Classifier/images/interface.png)    

**Classification Response**
![Response](https://github.com/stevenpham1996/Data-Science/blob/8d9f5adbd98149f99f112c954cdf66f04ef99d36/Disaster_Respones_Classifier/images/Screenshot_1.png)  
  

# Dependencies

```
Numpy
Pandas
Scikit-learn
xgboost
NLTK
regex
sqlalchemy
fastAPI
```  


# Installation   

in the virtual environment, clone the repository :
```https://github.com/stevenpham1996/Data-Science/tree/main/Disaster_Respones_Classifier```

To install the packages, run the following command:

`pip install -r requirements.txt`  

  
# Running the application  

You can run the following commands in the project's directory to set up the database, train the model and save the model.

To run ETL pipeline for cleaning data and store the processed data in the database:

Run the below command from the terminal:
- `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
  
To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file:

Run the below command from the terminal:
- `python model/ada_classifier.py data/DisasterResponse.db models/classifier.pkl`
  
Run the following command in the app's directory to run your web app:
- `python app.py`

Open a browser and go to http://127.0.0.1:8000/. You can input any message and see the results.
  

# Files  

app/templates/*: templates/html files for web application

data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database.

models/train_classifier.py: Machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file.

app/run.py: Launch the web app to classify disaster tweet messages.  
  

# Instructions:  

Run the below command from the terminal-
- `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file:

Run the below command from the terminal-
- `python model/ada_classifier.py data/DisasterResponse.db models/classifier.pkl`

Run the following command in the app's directory to run your web app-
- `python app.py`

Open a browser and go to http://127.0.0.1:8000/. You can input any message and see the results.


# Files

`app/templates/*`: templates/html files for web application

`data/process_data.py`: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

`models/train_classifier.py`: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

`app/run.py`: This file is used to launch the web app to classify disaster tweet messages.
    
  
# Acknowledgements

Thank you to Figure Eight for providing the dataset of this project.

