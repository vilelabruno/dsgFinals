import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

train = pd.read_csv('train_tracking.csv')
df = pd.DataFrame()
train['products'] = train['products'].fillna('["null"]')
for line in train['products']:
    jj = json.loads(line)
    df = pd.concat([jj, df])

df.to_csv('jj.csv', index=False)
