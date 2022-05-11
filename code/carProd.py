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
        fff = 0
        maxx = 0
        minn = 0
        stdd = 0
        if ll[0] != 'null':
            fff = ll[0]['rate']/ll[0]['rvoter']
            minn = ll[0]['rate']/ll[0]['rvoter']
            maxx = ll[0]['rate']/ll[0]['rvoter']
            for j in range(0, len(ll)):
                if ll[j]['price'] > maxx:
                    maxx = ll[0]['rate']/ll[0]['rvoter']
                if ll[j]['price'] < minn:
                    minn = ll[0]['rate']/ll[0]['rvoter']

                avgg += ll[0]['rate']/ll[0]['rvoter']
            
            avgg = avgg/len(ll)
            df['avgRateCar'][df['index'] == i] = avgg 
            df['maxRateCar'][df['index'] == i] = minn 
            df['minRateCar'][df['index'] == i] = maxx 
            df['fRateCar'][df['index'] == i] = fff 

    return df

train = pd.read_csv('train_tracking.csv')
df = pd.DataFrame()
train['carproducts'] = train['carproducts'].fillna('["null"]')
train = train.reset_index()
train['avgRateCar'] = 0
train['maxRateCar'] = 0
train['fRateCar'] = 0
train['minRateCar'] = 0
train = trat_json(train)
train.to_csv('train_trat1', index=False)
#df.to_csv('jj.csv', index=False)
