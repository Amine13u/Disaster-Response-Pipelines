import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''
    This function Load the dataset from the SQLite db
    
    Input : database_filepath (str) : Path to the SQLite db
    
    Outputs : X (dataframe) : Dataframe containing the feature variable
              Y (dataframe) : Dataframe containing the target variable 
              category_names (list) : List containing the categories names
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesCategories', engine)
    # Define feature and target variables X and Y
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, Y, category_names

def tokenize(text):
    '''
    This function tokenize the text
    
    Input : text (str) : Original message text
    
    Output : clean_tokens (list) : List of tokens cleaned
    '''
    # Detect urls and replace them with a placeholder string
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Lemmatize, normalize and clean the tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    This function build a ML pipeline
    
    Input : None
    
    Output : cv (class) : A GridSearchCV
    '''
    # Build a machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Setting parameters to test
    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }

    # Testing parameters
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function build a ML pipeline
    
    Inputs : model (class) : A ML model
             X_test (dataframe) : A feature test Dataframe 
             Y_test (dataframe) : A target test Dataframe 
             category_names (list) : A list of the categories
    
    Output : None
    '''
    y_pred = model.predict(X_test)
    
    for column in category_names:
        print('Category : ' + column)
        print(classification_report(y_test[column], y_pred_df[column]))
    

def save_model(model, model_filepath):
    '''
    This function saves the model to a pickle file
    
    Inputs : model (class) : A ML model
             model_filepath (string) : Path to the pickle file
            
    Output : None
    '''
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()


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