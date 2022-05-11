import os
import json
import numpy as np
import pandas as pd
import time
from pandas.io.json import json_normalize

def trat_json(df):

    start_time = time.time()
    i = 0
    for line in df['dproducts']:
        ll = eval(line)
        avgg = 0
        soma = 0
        maxx = 0
        minn = 0
        stdd = 0
        if ll[0] != 'null':
            df['countDProd'][df['index'] == i] = len(ll)

    return df

train = pd.read_csv('train_tracking.csv')
df = pd.DataFrame()
train['products'] = train['products'].fillna('["null"]')
train = train.reset_index()
train['countDProd'] = 0
train = trat_json(train)
train.to_csv('train_trat1', index=False)
#df.to_csv('jj.csv', index=False)
