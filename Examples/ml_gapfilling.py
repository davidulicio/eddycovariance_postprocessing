# -*- coding: utf-8 -*-
"""
Machine Learning Methods to gapfilling timeseries
David Trejo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, VotingRegressor

#%%
def gapfilling(df1, df2, method, test_size):
    """
    Machine learning techniques to gapfilling timeseries using one or multiple 
    features as input. 

    Parameters
    ----------
    df1 : DataFrame
        DataFrame with its columns as the features for training and prediction.
        The index is the timestamp of the timeseries.
    df2 : DataFrame, Series
        Values of the variable to be predicted. Index must be the timestamp of 
        the timeseries.
    method : string
        String of the method to apply. The methods available are linear
        regression, 

    Returns
    -------
    prediction : Series.
        Series of the the predicted values using the method selected and given
        inputs.
    
    """
    date = df1.index.intersection(df2.index)
    df1 = df1[date]; df2 = df2[date]
    valid_data = ~((df1.isna().sum(axis=1)>0) or (df2.isna()))
    df1 = df1[valid_data]; df2 = df2[valid_data]
    X_train, X_test, y_train, y_test = train_test_split(df1, df2, test_size=0.2, random_state=42)
    if method == 'linear':
        reg = linear_model.LinearRegression()
    elif method == 'svm':
        reg = svm.SVR()
    elif method == 'kneighbors':
        reg = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform')
    elif method == 'scalar':
        reg = make_pipeline(StandardScaler(),linear_model.SGDRegressor(max_iter=1000, tol=1e-3))
    elif method == 'tree':
        reg = tree.DecisionTreeRegressor()
    elif method == 'bagging':
        reg = BaggingRegressor(random_state=42)
    reg.fit(X_train, y_train)
    score_train = str(round(reg.score(X_train, y_train), 2))
    score_test = str(round(reg.score(X_test, y_test), 2))

    print("""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%""")
    print("The gapfilling method selected is: "+ method)
    print('The Test size is: '+str(test_size) + "The Train size is: " +
          str(1-test_size))
    print("The Train Sample Score for the selected method is: "+ score_train)
    print("The Test Sample Score for the selected method is: "+ score_test)
    prediction = reg.predict(df1)
    prediction = pd.Series(prediction, index=df1.index)
    df3 = df2.copy()
    df3[df3.isna()] = prediction[df3.isna()]
    # Figure
    fig, axs = plt.subplots(3, 1, figsize=(8,8), sharex=True)
    axs[0].plot(df2)
    axs[0].set_title('Raw Data')
    axs[1].plot(prediction)
    axs[1].set_title('Model')
    axs[2].plot(df3)
    axs[2].set_title('Filled Data')
    
    return prediction
