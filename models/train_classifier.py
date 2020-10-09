import sys
# import libraries
import pandas as pd
import numpy as np
# for sql lite db
from sqlalchemy import create_engine
# NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# libraries for NLP feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# library for multi target classification
from sklearn.multioutput import MultiOutputClassifier
# model evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# pickle library to save machine learning model
import pickle

def load_data(database_filepath):
    '''
    Function to load data from sql lite db.
    
    ARGS:
    database_filepath: path to sqlite db file
    
    OUTPUT:
    X: message column
    Y: 36 output categories
    category_names: name of output categories
    
    '''
    # load data from database
    
    engine = create_engine('sqlite:////home/workspace/' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    
    df = df.drop(columns = 'original')
    # drop Null value
    df = df.dropna()
    
    X = df["message"].values
    Y = df.drop(["id", "message", "genre"], axis=1).values
    category_names = df.drop(["id", "message", "genre"], axis=1).columns
    return X, Y, category_names


def tokenize(text):
    '''
    Function to tokenize words within a message.
    
    ARGS:
    text: message to be word tokenized
    
    OUTPUT:
    clean_tokens: cleaned word tokens of message
    
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    Function to build model pipeline with feature extraction and estimator.
    
    ARGS:
    None
    
    OUTPUT:
    pipeline: built model
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('MultiOutput-clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
#         'tfidf__use_idf': (True, False),
        'MultiOutput-clf__estimator__criterion': ["gini", "entropy"],     
        'MultiOutput-clf__estimator__n_jobs':[-1]
        
        }
    cv = GridSearchCV(pipeline, parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to print out evaluation of model on test data.
    
    ARGS:
    model: trained model
    X_test: messages test data
    Y_test: output categories test data
    category_names: name of output categories
    
    OUTPUT:
    Classification_report for each category
    
    '''
    # predict on test data
    Y_pred = model.predict(X_test)
    # Todo:  iterating through the category columns
    index = 0
    for category in category_names:
        print("output category in column {}: {}".format(index, category))
        evaluation_report = classification_report(Y_test[:,index], Y_pred[:,index])
        index += 1
        print(evaluation_report)


def save_model(model, model_filepath):
    '''
    Function to export model as a pickle file.
    
    ARGS:
    model: trained model
    model_filepath: path and filename to save pickle file of trained model
    
    
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()