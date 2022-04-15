#!/usr/bin/env python
# coding: utf-8
# Author: Tanmay C. Shidhore

###################################################################
#                       Main Code
###################################################################


# Import packages necessary for the script
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from random import random
from matplotlib import pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from pdb import set_trace as keyboard
import Create_State_CSVs
import Plot_Predict_State_CSVs

# This code needs the 'EV_sales_final.csv' file

# Note: The 'EV_Sales_final.csv file should be in the same folder as the codes
# Code based on https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model_exog(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        #print(model_fit.summary())
        yhat = model_fit.forecast(1,alpha=0.05)[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    #keyboard()
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return rmse

# evaluate an ARIMA model with exogenous variables for a given order (p,d,q)
def evaluate_arima_model(X, arima_order, exogenous):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    train_eg, test_eg = exogenous[0:train_size,:], exogenous[train_size:,:]
    history = [x for x in train]
    exogen = [y for y in train_eg]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order, exog=exogen)
        model_fit = model.fit()
        #print(model_fit.summary())
        yhat = model_fit.forecast(1,alpha=0.05,exog=test_eg[t,:])[0]
        predictions.append(yhat)
        history.append(test[t])
        exogen.append(test_eg[t,:])
    # calculate out of sample error
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return rmse
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models_exog(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model_exog(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return best_cfg

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values, exogenous):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order, exogenous)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return best_cfg

# For a give state, find best ARIMA models for all exogenous variabbles, 
# and write output CSV file with exogenous variable predictions.
# This will be used to predict EV sales.
def STATE_ARIMA_EXOG(State_abbrv):

    filename = './' + State_abbrv + '_EV_sales_data_final.csv'
    filename2 = './' + State_abbrv + '_Exogenous_ARIMA_prediction_data.csv'
    State_df = pd.read_csv(filename)
    State_df.info()
    State_df.head()
    State_df.keys()
    
    State_df = State_df.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
    State_df['Year'] = pd.to_datetime(State_df.Year, format='%Y')
    State_df.set_index('Year',inplace=True)
    State_df.rename(columns={'Gasoline Price':'Gasoline_Price','Median Income':'Median_Income','Lithium Price':'Lithium_Price'},inplace=True)
    # Range of (p,d,q)
    p_values = range(0, 3)
    d_values = range(0, 3)
    q_values = range(0, 3)
    # Supresses warnings
    warnings.filterwarnings("ignore")
    
    # ARIMA model for Gasoline prices
    
    best_cfg_GP = evaluate_models_exog(State_df['Gasoline_Price'].values, p_values, d_values, q_values)

    p = best_cfg_GP[0]
    d = best_cfg_GP[1]
    q = best_cfg_GP[2]

    ARIMA_GP_model = ARIMA(State_df['Gasoline_Price'],order=(p,d,q))
    ARIMA_GP_model_fit = ARIMA_GP_model.fit()
    print("ARIMA Model for Gasoline Prices\n")
    print(ARIMA_GP_model_fit.summary())
    
    # Forecast of gasoline prices
    pred_GP = ARIMA_GP_model_fit.forecast(15, alpha=0.05)  # 95% conf
    
    # Create a dataframe to store exogenous variable predictions
    Exog_df = pd.DataFrame(pred_GP)
    Exog_df.rename(columns={'predicted_mean':'Gasoline_Price_predicted'},inplace=True)
    
    
    # ARIMA model for Median Household Income
    
    best_cfg_MI = evaluate_models_exog(State_df['Median_Income'].values, p_values, d_values, q_values)
    
    p = best_cfg_MI[0]
    d = best_cfg_MI[1]
    q = best_cfg_MI[2]
    
    ARIMA_MI_model = ARIMA(State_df['Median_Income'],order=(p,d,q))
    ARIMA_MI_model_fit = ARIMA_MI_model.fit()
    print("ARIMA Model for Median Household Income\n")
    print(ARIMA_MI_model_fit.summary())
   
    # Forecast Median Household income
    pred_MI = ARIMA_MI_model_fit.forecast(15, alpha=0.05)  # 95% conf
    
    # Append to dataframe with exogenous variable predictions
    Exog_df = pd.concat([Exog_df,pd.DataFrame(pred_MI)], axis=1)
    Exog_df.rename(columns={'predicted_mean':'Median_Income_predicted'},inplace=True)
    Exog_df.head()
    
    
    # ARIMA model for Lithium Price
    
    best_cfg_LI = evaluate_models_exog(State_df['Lithium_Price'].values, p_values, d_values, q_values)
    
    
    p = best_cfg_LI[0]
    d = best_cfg_LI[1]
    q = best_cfg_LI[2]
    
    ARIMA_LP_model = ARIMA(State_df['Lithium_Price'],order=(p,d,q))
    ARIMA_LP_model_fit = ARIMA_LP_model.fit()
    print(ARIMA_LP_model_fit.summary())
    
    # Forecast Lithium price
    pred_LP = ARIMA_LP_model_fit.forecast(15, alpha=0.05)  # 95% conf

    # Append to dataframe with exogenous variable predictions
    Exog_df = pd.concat([Exog_df,pd.DataFrame(pred_LP)], axis=1)
    Exog_df.rename(columns={'predicted_mean':'Lithium_Price_predicted'},inplace=True)
    Exog_df.head()
   
    # Write out all exogenous predictions as a CSV
    Exog_df.to_csv(filename2)

def STATE_ARIMA_EV_SALES(State_abbrv):
    filename = './' + State_abbrv + '_EV_sales_data_final.csv'
    filename2 = './' + State_abbrv + '_Exogenous_ARIMA_prediction_data.csv'
    filename3 = './' + State_abbrv + '_EV_sales_prediction_data.csv'
    State_df = pd.read_csv(filename)
    State_df.info()
    State_df.head()
    State_df.keys()

    # Cleanup columns and set date as index

    State_df = State_df.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
    State_df['Year'] = pd.to_datetime(State_df.Year, format='%Y')
    State_df.set_index('Year',inplace=True)
    State_df.rename(columns={'Gasoline Price':'Gasoline_Price','Median Income':'Median_Income','Lithium Price':'Lithium_Price'},inplace=True)
    # Range of (p,d,q)
    p_values = range(0, 3)
    d_values = range(0, 3)
    q_values = range(0, 3)

    warnings.filterwarnings("ignore")
    best_cfg_EV = evaluate_models(State_df['Total'].values, p_values, d_values, q_values, np.column_stack([State_df['Gasoline_Price'].values,State_df['Median_Income'].values,State_df['Lithium_Price'].values]))
    p = best_cfg_EV[0]
    d = best_cfg_EV[1]
    q = best_cfg_EV[2]
    
    ARIMA_model = ARIMA(State_df['Total'],order=(p,d,q),exog=np.column_stack([State_df['Gasoline_Price'],State_df['Median_Income'],State_df['Lithium_Price']]))
    ARIMA_model_fit = ARIMA_model.fit()
    print("ARIMA model for EV sales")
    print(ARIMA_model_fit.summary())

    # Forecast EV sales
    Exog_df = pd.read_csv(filename2)
    pred_sales = ARIMA_model_fit.forecast(15,exog=np.column_stack([Exog_df['Gasoline_Price_predicted'],Exog_df['Median_Income_predicted'],Exog_df['Lithium_Price_predicted']]), alpha=0.05)  # 95% conf

    Predicted_df = pd.DataFrame(pred_sales)
    Predicted_df.rename(columns={'predicted_mean':'Sales_predicted'},inplace=True)
    # Write out predictions as CSV
    Predicted_df.to_csv(filename3)
    
# This is a list two-letter acronyms of states within MISO's footprint
state_list = ['AR','IA','IL','IN','LA','MI','MN','MO','MS','ND','WI']

Create_State_CSVs.CREATE_CSV()

for state in state_list:
    STATE_ARIMA_EXOG(state)
    STATE_ARIMA_EV_SALES(state)

Plot_Predict_State_CSVs.PLOT_DATA()