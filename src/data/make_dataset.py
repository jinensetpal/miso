#!/usr/bin/env python3

import pandas as pd
def dataset(csv):
    """Makes the original dataset and removes data that do not match the year for 
       the information we plan to make.
    Args:
      csv file that we plan to subset
    Returns:
      Refined a dataset with the data we need for analysis.
    """
    df = pd.read_csv("state_year.csv")
    df = df[df['Year']>2010]
    df = df[df['Year']<2019]
    df =df.reset_index()
    return df
    

def mergeWithLithiumPrice(df,csvFileName):
    """Combines the original dataset with another csv file(that has lithium price)
    Merge based on year and its lithium price for that year
    Args:
      the original dataset which we plan to merge with the csv file
      csv file that we plan to merge into the original dataframe
    Returns:
      Refined a dataset with both dataset alreadu merge
    """
    df1 = pd.read_csv(csvFileName)
    df1.head()
    allyear = df['Year'].tolist()
    years = df1['Year'].tolist()
    prices = df1['Price'].tolist()
    temp = []
    for i in allyear:
        if i in years:
            index = years.index(i)
            price = prices[index]
            temp.append(price)t
        else:
            temp.append('No dataset for lithium price')
    states =['ND','MN','IA','WI','MI','IL','MO','AR','LA','MS']
    df.loc[df['State'].isin(states)]
    statesPerYearInorder = []
    df['Lithium Price'] = temp
    df =df.reset_index()
    return df



def mergeWithEVSalesData(df,csvFileName):
    """Combines the original dataset with another csv file(year and ev sales)
        Merge based on states and year
    Args:
      the original dataset which we plan to merge with the csv file
      csv file that we plan to merge into the original dataframe
    Returns:
      Refined a dataset with both dataset alreadu merge
    """
    state_year = pd.read_csv(csvFileName)
    df = pd.merge(state_year, df, on = ["State", "Year"])
    return df


if __name__ == '__main__':
    df = dataset("state_year.csv")
    mergeWithLithiumPrice(df,"li_price.csv")
    mergeWithEVSalesData(df,"us_ev_salesdata.csv")