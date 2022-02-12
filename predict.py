import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import re
import json
import warnings
warnings.filterwarnings("ignore")


def region_list_gen(df):
    df['regions'] = df['regions'].apply(lambda row: row.lower())
    lists = df['regions'].unique().tolist()
    with open('regions_list.json','w', encoding='utf-8') as f:
        json.dump(lists, f, ensure_ascii=False,indent=4)
    return lists, df

def selecting_region(df,region):
    """
    this function will 
    """
    df = df.loc[df['regions']==region]
    df = df.T
    df.dropna(inplace=True)
    df = df.reset_index()
    return df

def prediction_model(df):
    x = df.iloc[:, 0].values.reshape(-1,1)
    y = df.iloc[:, 1].values.reshape(-1,1)
    model = LinearRegression().fit(x,y)
    print(accuracy_score(x, y, normalize=False))
    return model

def prediction(model, year):
    return int(model.coef_[0][0] * year + model.intercept_[0])


def main():
    region = input("Please input the region name: ").lower()
    year = int(input("Please input the year to predict: "))
    df = pd.read_csv('population.csv')
    lists, df = region_list_gen(df)
    if region in lists:
        df = selecting_region(df, region)
        model = prediction_model(df)
        result = prediction(model,year)
        print(f"\n Result: {region.upper()} population in {year} will be {result:,d}")
    else:
        print('kindly check available region name and thier spelling from regions_list.json')
    
if __name__ == "__main__":
    main()