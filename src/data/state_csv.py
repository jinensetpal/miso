#!/usr/bin/env python
# coding: utf-8
# Author: Tanmay C. Shidhore

def CREATE_CSV():
    # Import necessary packages
    import pandas as pd
    
    # Importing the compiled csv file and print a summary
    filename = './EV_sales_data_final.csv'
    df = pd.read_csv(filename)
    df.info()
    
    # This is a list two-letter acronyms of states within MISO's footprint
    state_list = ['AR','IA','IL','IN','LA','MI','MN','MO','MS','ND','WI']
    
    # Write out a csv file for each specific state
    for state in state_list:
        State_df = df.loc[df['State'] == state]
        State_df = State_df.drop(columns=['State'])
        State_df.to_csv('./' + state + '_EV_sales_data_final.csv')

