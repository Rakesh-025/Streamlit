# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_excel(r"C:\360digitmg Project\Data Science Coaching Competitors final draft (1).xlsx")

dataset.Maintenance_cost = dataset.Maintenance_cost.astype('int64')
dataset.dtypes


dataset.Marketing_cost = dataset.Marketing_cost.astype('int64')
dataset.dtypes

dataset.Profit_Margin = dataset.Profit_Margin.astype('int64')
dataset.dtypes

dataset.Debentures = dataset.Debentures.astype('int64')
dataset.dtypes

dataset.Duration_of_coaching_in_Hours = dataset.Duration_of_coaching_in_Hours.astype('int64')
dataset.dtypes

# dataset['Experience'].fillna(0, inplace=True)

dataset['Maintenance_cost'].fillna(dataset['Maintenance_cost'].median(), inplace=True)

dataset['Marketing_cost'].fillna(dataset['Marketing_cost'].mean(), inplace=True)

dataset['Profit_Margin'].fillna(dataset['Profit_Margin'].mean(), inplace=True)

dataset['Debentures'].fillna(dataset['Debentures'].mean(), inplace=True)


dataset['Duration_of_coaching_in_Hours'].fillna(dataset['Duration_of_coaching_in_Hours'].median(), inplace=True)



df= dataset[['Maintenance_cost','Marketing_cost','Profit_Margin','Debentures','Duration_of_coaching_in_Hours']]
#  X = dataset.iloc[:, 2:16]
'''
#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'elever':11, 'twelve':12, 'nan'= null, 'zero':0, 0: 0}
    return word_dict[word] 
def convert_to_int(word):
    word_dict = {'Beginner':1, 'Intermediate':2, 'Higher':3, 0: 0}
    return word_dict[word]
'''


# df['Level_of_Course'] = df['Level_of_Course'].apply(lambda df : convert_to_int(df))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

#import statsmodels.formula.api as smf # for regression model  
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(df, y)

# Saving model to disk
pickle.dump(regressor, open('model_Prediction.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_Prediction.pkl','rb'))
print(model.predict([[2220, 945, 64,2,10]]))
