import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

## Save the pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
##Load the pickle file
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)     
    


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]  ##Inside the loop, the code extracts the current machine learning model by using the list() function to get the i-th element of the list of models and assigns it to the model variable. 
            #The list() function is used here to convert the dictionary values to a list so that we can access them by index.

            #model.fit(X_train, y_train)  # Train model
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            '''
            In the given code snippet, ** is used to unpack a dictionary. Specifically, **gs.best_params_ is used to unpack the dictionary gs.best_params_ and pass its contents as keyword arguments to the set_params method of the model object.
            The best_params_ attribute of the GridSearchCV object gs returns a dictionary of the best hyperparameters found during the grid search. The double asterisks (**) are used to unpack this dictionary into a series of keyword arguments, which can then be passed to the set_params method of the model object.
            By using ** to unpack the dictionary, each key-value pair in the dictionary becomes a separate keyword argument that is passed to the method. This allows you to dynamically set the hyperparameters of the model using the best parameters found during the grid search.   
            '''
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            """
            This code snippet creates an empty dictionary called report. It then adds a new key-value pair to the dictionary where the key is the ith key in a dictionary called models (which is not shown in the code snippet), and the value is test_model_score.
            To break it down further:
            models.keys() returns a list of all the keys in the models dictionary.
            list(models.keys())[i] gets the ith key in that list. The list() function is used to convert the keys from a dictionary view object to a list so that we can access them by index.
            test_model_score is a variable that presumably contains some kind of test score for the model.
            So, the overall effect of this code is to create a dictionary called report with keys corresponding to the names of the models in the models dictionary, and values corresponding to their test scores. The i variable determines which key in models to use as the key in report.
            """

        return report

    except Exception as e:
        raise CustomException(e, sys)
    


    '''
    # Define a dictionary
    params = {'param1': 10, 'param2': 20, 'param3': 30}

    # Define a function that takes three parameters
    def my_func(param1, param2, param3):
        print(f'param1={param1}, param2={param2}, param3={param3}')

    # Call the function using the dictionary as keyword arguments
    my_func(**params)

    In this example, the params dictionary contains three key-value pairs that correspond to the three parameters expected by the my_func function. By using the double asterisk (**) operator to unpack the dictionary, we can pass its contents as keyword arguments to the function.

    When we call my_func(**params), the dictionary contents are unpacked and passed as separate keyword arguments to the function. The output of the function call will be:



    o/p is
    param1=10, param2=20, param3=30

    This demonstrates how the double asterisk (**) operator can be used to unpack a dictionary and pass its contents as keyword arguments to a function.
    '''
   