import os
import json
import numpy as np
import pandas as pd
import time
from pandas.io.json import json_normalize

def trat_json(df):

    start_time = time.time()
    i = 0
    for line in df['carproducts']:
        ll = eval(line)
        avgg = 0
        soma = 0
        maxx = 0
        minn = 0
        stdd = 0
        if ll[0] != 'null':
            minn = ll[0]['price']
            maxx = ll[0]['price']
            for i in range(0, len(ll)):
                if ll[i]['price'] > maxx:
                    maxx = ll[i]['price']
                if ll[i]['price'] < minn:
                    minn = ll[i]['price']

                avgg += ll[i]['price']
            soma = avgg
            avgg = avgg/len(ll)
            stdd = (((soma - avgg)**2)/len(ll))**(1/2)
            df['avgProdPrice'][df['index'] == i] = avgg 
            df['minProdPrice'][df['index'] == i] = minn 
            df['maxProdPrice'][df['index'] == i] = maxx 

    return df

train = pd.read_csv('train_tracking.csv')
df = pd.DataFrame()
train['products'] = train['products'].fillna('["null"]')
train = train.reset_index()
train['countDProd'] = 0
train = trat_json(train)
train.to_csv('train_trat1', index=False)
#df.to_csv('jj.csv', index=False)
