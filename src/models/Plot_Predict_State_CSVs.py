#!/usr/bin/env python
# coding: utf-8
# Author: Tanmay C. Shidhore

# Import packages necessary for the script
import pandas as pd
import numpy as np
from pdb import set_trace as keyboard
from matplotlib import pyplot as plt

def PLOT_DATA():
    # This is a list two-letter acronyms of states within MISO's footprint
    state_list = ['AR','IA','IL','IN','LA','MI','MN','MO','MS','ND','WI']
    
    # Range of years over which data is available.
    Year = np.arange(2019,2034,1).astype(int)
    Total_sales = np.zeros(len(Year),dtype=int)
    
    # Compile the sales predictions for each state
    for state in state_list:
        filename3 = './' + state + '_EV_sales_prediction_data.csv'
        df1 = pd.read_csv(filename3)
        Total_sales += df1.Sales_predicted.values.astype(int)
    
    # Importing the original csv
    filename = './EV_sales_data_final.csv'
    Year_org = np.arange(2011,2019,1).astype(int)
    Total_sales_org = np.zeros(len(Year_org),dtype=int)
    for state in state_list:
        filename = './' + state + '_EV_sales_data_final.csv'
        df = pd.read_csv(filename)
        Total_sales_org += df.Total.values.astype(int)
    
    # creating the bar plot showing EV predictions
    plt.figure(figsize = (10, 10))
    plt.bar(Year_org,Total_sales_org, color ='maroon',width = 0.4,label='Historical')
    plt.bar(Year[0:11],Total_sales[0:11], color ='green',width = 0.4,label='Predicted')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Year",fontsize=16)
    plt.ylabel("EV Sales",fontsize=16)
    plt.title("Total EV Sales prediction",fontsize=16)
    plt.legend(loc='best',fontsize=16)
    plt.show()



