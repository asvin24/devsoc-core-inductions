import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(file_name,test_size,random_state=None): 

    '''the script assumes what the features in the given dataset are 
    and thus it may not work with any other datasets that 
    doesn't have the same set of features present in 'dataset.csv' '''   


    df=pd.read_csv(file_name)
    gen=LabelEncoder()
    fam_hist=LabelEncoder()
    favc=LabelEncoder()
    caec=LabelEncoder()
    smoke=LabelEncoder()
    scc=LabelEncoder()
    calc=LabelEncoder()
    mtrans=LabelEncoder()
    obesity_level=LabelEncoder()

    df['Gender']=gen.fit_transform(df['Gender'])
    df['family_history_with_overweight']=fam_hist.fit_transform(df['family_history_with_overweight'])
    df['FAVC']=favc.fit_transform(df['FAVC'])
    df['CAEC']=caec.fit_transform(df['CAEC'])
    df['SMOKE']=smoke.fit_transform(df['SMOKE'])
    df['SCC']=scc.fit_transform(df['SCC'])
    df['CALC']=calc.fit_transform(df['CALC'])
    df['MTRANS']=mtrans.fit_transform(df['MTRANS'])
    df['NObeyesdad']=obesity_level.fit_transform(df['NObeyesdad'])


    X=df.drop('NObeyesdad',axis='columns').values
    y=df['NObeyesdad'].values

    if test_size==0:
        return X,y

    return train_test_split(X,y,test_size=test_size,random_state=random_state)