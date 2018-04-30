# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Load the dataset
in_file = "D:/Projects/machine-learning-master/machine-learning-master/projects/titanic_survival_exploration/titanic_survival_exploration/titanic_data.csv"
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
display(full_data.head())

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(data.head())

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print(accuracy_score(outcomes[:5], predictions))

def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data.iterrows():
        
        # Predict the survival of 'passenger'
        predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_0(data)
print(accuracy_score(outcomes, predictions))

vs.survival_stats(data, outcomes, 'Sex')

def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if passenger['Sex'] == "female":
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)
print(accuracy_score(outcomes, predictions))

vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])

def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if passenger['Sex'] == "female":
            predictions.append(1)
        else:
            if passenger['Age'] <= 10:
               predictions.append(1)
            else: 
               predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)
print(accuracy_score(outcomes, predictions))

vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Age < 18"])

vs.survival_stats(data, outcomes, 'Pclass', ["Sex == 'male'", "Age < 18"])

def predictions_3(data):
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if passenger['Sex'] == "female":
            predictions.append(1)
        else:
            if passenger['Age'] <= 10:
               predictions.append(1)  
            else:
               if passenger['Fare'] >= 120 and passenger['Fare'] < 150:
                  predictions.append(1)
               else:
                if passenger['Fare'] > 120:
                  predictions.append(0)
                else:    
                  if passenger['Pclass'] > 1:
                     predictions.append(0)
                  else:    
                      if passenger['SibSp'] > 1:
                          predictions.append(0)
                      else:
                          if passenger['Parch'] > 1:
                              predictions.append(0)
                          else:
                            if passenger['Age'] > 20 and passenger['Age'] < 30:
                              predictions.append(1)          
                            else:
                              if passenger['Age'] > 30 and passenger['Age'] < 40 and passenger['Fare'] < 40:
                                  predictions.append(1)
                              else:
                                  if passenger['Age'] > 40 and passenger['Age'] < 50 and passenger['SibSp'] == 10:
                                     predictions.append(1)
                                  else:   
                                     predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)
print(accuracy_score(outcomes, predictions))

#vs.survival_stats(data, outcomes, 'SibSp', ["Sex == 'male'","Age > 40","Age < 50" , "Fare < 120" , "Pclass == 1","SibSp == 1" ,"Parch <= 1"])


"""
file = open("D:/Projects/machine-learning-master/machine-learning-master/projects/titanic_survival_exploration/titanic_survival_exploration/titanic_data.csv","r") 
line = file.readline()
print(line)
file.close()
""" 
