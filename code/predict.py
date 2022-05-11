
import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn import *

def transform_time(df_train):
    df_train['time'], df_train['mili'] = df_train['duration'].str.split('.', 1).str
    df_train['days'], df_train['time'] = df_train['time'].str.split(' days ', 1).str
    df_train['hour'], df_train['min'], df_train['sec'] = df_train['time'].str.split(':', 2).str
    df_train['seconds'] = df_train['sec'].astype(int) + (df_train['min'].astype(int)*60) + (df_train['hour'].astype(int)*3600) + (df_train['days'].astype(int)*86400)
    df_train = df_train.drop(columns = ['days', 'hour', 'min', 'sec', 'mili', 'time'])
    return df_train

# scala label
train = pd.read_csv('train_tracking.csv')
test = pd.read_csv('test_tracking.csv')

target = pd.read_csv("train_session.csv")
prdCat = pd.read_csv("productid_category.csv")
    
print("Transforming the data...")

# Session Count
f_train = train.groupby('sid')['type'].count().reset_index()
f_train.columns = ['sid', 'session_count']
f_test = test.groupby('sid')['type'].count().reset_index()
f_test.columns = ['sid', 'session_count']

nrow_train = train.shape[0]
merge = pd.concat([train, test])

merge = transform_time(merge)

# One-Hot Type
typ_columns = []
aux = merge[['sid','type']]
ohe = pd.get_dummies(aux['type'])
ohe_columns = ohe.columns
for j in ohe_columns:
    aux['num_'+j] = ohe[j]
    typ_columns.append('num_'+j)
aux.drop('type', axis=1, inplace=True)
aux = aux.groupby('sid').sum().reset_index()
f_train = pd.merge(f_train, aux, on='sid', how='left', sort=False)
f_test  = pd.merge(f_test, aux, on='sid', how='left', sort=False)

for col in typ_columns:
    f_train['perc_'+col] = f_train[col] / f_train['session_count']
    f_test['perc_'+col]  = f_test[col] / f_test['session_count']

# One-Hot Type Simplified
aux = merge[['sid','type_simplified']]
ohe = pd.get_dummies(aux['type_simplified'])
ohe_columns = ohe.columns
for j in ohe_columns:
    aux['num_'+j] = ohe[j]
aux.drop('type_simplified', axis=1, inplace=True)
aux = aux.groupby('sid').sum().reset_index()
f_train = pd.merge(f_train, aux, on='sid', how='left', sort=False)
f_test  = pd.merge(f_test, aux, on='sid', how='left', sort=False)

# Treat integer columns
int_columns = ['nb_query_terms','rcount','quantity','orcount','seconds']
for col in int_columns:
    encoding = merge.groupby('sid')[col].mean().reset_index()
    encoding.columns = ['sid',col+'_mean']
    f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
    f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in int_columns:
    encoding = merge.groupby('sid')[col].std().reset_index()
    encoding.columns = ['sid',col+'_std']
    f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
    f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in int_columns:
    encoding = merge.groupby('sid')[col].max().reset_index()
    encoding.columns = ['sid',col+'_max']
    f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
    f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in int_columns:
    encoding = merge.groupby('sid')[col].min().reset_index()
    encoding.columns = ['sid',col+'_min']
    f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
    f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in int_columns:
    encoding = merge.groupby('sid')[col].sum().reset_index()
    encoding.columns = ['sid',col+'_sum']
    f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
    f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)

#Treat other numerical columns
num_columns = ['opn','pn']
for col in num_columns:
    encoding = merge.groupby('sid')[col].max().reset_index()
    encoding.columns = ['sid',col+'_max']
    f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
    f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in num_columns:
    encoding = merge.groupby('sid')[col].min().reset_index()
    encoding.columns = ['sid',col+'_min']
    f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
    f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)

# Treat categorical columns
cat_columns = ['query','sku','sname','oquery','offerid','stype','facets']
rst_columns = []
for col in cat_columns:
    encoding = merge.groupby(col)['type'].count().reset_index()
    encoding.columns = [col, col+'_frequency']
    rst_columns.append(col+"_frequency")
    merge = pd.merge(merge, encoding, on=col, how='left', sort=False)

for col in rst_columns:
    encoding = merge.groupby('sid')[col].max().reset_index()
    encoding.columns = ['sid',col+'_max']
    f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
    f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in rst_columns:
    encoding = merge.groupby('sid')[col].min().reset_index()
    encoding.columns = ['sid',col+'_min']
    f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
    f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)

merge['ff'][merge['ff'] == True]  = 1
merge['ff'][merge['ff'] == False] = 0
encoding = merge.groupby('sid')['ff'].sum().reset_index()
encoding.columns = ['sid','ff_sum']
f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)

f_train = pd.merge(f_train, target, on='sid', how='left', sort=False)
target = f_train['target']

# Read meta features...
sid = f_test['sid']
f_train.drop(["target"], axis=1, inplace=True)
f_test.drop("sid", axis=1, inplace=True)
f_train.drop("sid", axis=1, inplace=True)
print(f_train.columns)

# Training Session
print("Starting Kfolds...")
folds = KFold(n_splits=5, shuffle=True, random_state=159)

preds = np.repeat(0, len(f_test))
preds2 = np.repeat(0, len(f_test))
result = 0

train_features = ['session_count', 'num_ADD_TO_BASKET_CAROUSEL', 'num_ADD_TO_BASKET_LP',
       'num_ADD_TO_BASKET_LR', 'num_ADD_TO_BASKET_PA',
       'num_ADD_TO_BASKET_SHOW_CASE', 'num_CAROUSEL_x', 'num_LIST_PRODUCT_x',
       'num_PA_x', 'num_PRODUCT_CAROUSEL', 'num_PRODUCT_LP', 'num_PRODUCT_LR',
       'num_PRODUCT_PA', 'num_PRODUCT_SHOW_CASE',
       'num_PURCHASE_PRODUCT_CAROUSEL', 'num_PURCHASE_PRODUCT_LP',
       'num_PURCHASE_PRODUCT_LR', 'num_PURCHASE_PRODUCT_PA',
       'num_PURCHASE_PRODUCT_SHOW_CASE', 'num_PURCHASE_PRODUCT_UNKNOW_ORIGIN',
       'num_SEARCH_x', 'num_SHOW_CASE_x', 'perc_num_ADD_TO_BASKET_CAROUSEL',
       'perc_num_ADD_TO_BASKET_LP', 'perc_num_ADD_TO_BASKET_LR',
       'perc_num_ADD_TO_BASKET_PA', 'perc_num_ADD_TO_BASKET_SHOW_CASE',
       'perc_num_CAROUSEL', 'perc_num_LIST_PRODUCT', 'perc_num_PA',
       'perc_num_PRODUCT_CAROUSEL', 'perc_num_PRODUCT_LP',
       'perc_num_PRODUCT_LR', 'perc_num_PRODUCT_PA',
       'perc_num_PRODUCT_SHOW_CASE', 'perc_num_PURCHASE_PRODUCT_CAROUSEL',
       'perc_num_PURCHASE_PRODUCT_LP', 'perc_num_PURCHASE_PRODUCT_LR',
       'perc_num_PURCHASE_PRODUCT_PA', 'perc_num_PURCHASE_PRODUCT_SHOW_CASE',
       'perc_num_PURCHASE_PRODUCT_UNKNOW_ORIGIN', 'perc_num_SEARCH',
       'perc_num_SHOW_CASE', 'num_ADD_TO_BASKET', 'num_CAROUSEL_y',
       'num_LIST_PRODUCT_y', 'num_PA_y', 'num_PRODUCT', 'num_PURCHASE_PRODUCT',
       'num_SEARCH_y', 'num_SHOW_CASE_y', 'nb_query_terms_mean', 'rcount_mean',
       'quantity_mean', 'orcount_mean', 'seconds_mean', 'nb_query_terms_std',
       'rcount_std', 'quantity_std', 'orcount_std', 'seconds_std',
       'nb_query_terms_max', 'rcount_max', 'quantity_max', 'orcount_max',
       'seconds_max', 'nb_query_terms_min', 'rcount_min', 'quantity_min',
       'orcount_min', 'seconds_min', 'nb_query_terms_sum', 'rcount_sum',
       'quantity_sum', 'orcount_sum', 'seconds_sum', 'opn_max', 'pn_max',
       'opn_min', 'pn_min', 'query_frequency_max', 'sku_frequency_max',
       'sname_frequency_max', 'oquery_frequency_max', 'offerid_frequency_max',
       'stype_frequency_max', 'facets_frequency_max', 'query_frequency_min',
       'sku_frequency_min', 'sname_frequency_min', 'oquery_frequency_min',
       'offerid_frequency_min', 'stype_frequency_min', 'facets_frequency_min',
       'ff_sum']

oof_preds = pd.DataFrame()
lgb_params = {
    'learning_rate': 0.05,
    'application': 'binary',
    'max_depth': 5,
    'num_leaves': 128,
    'verbosity': -1,
    'metric': 'binary_logloss'
}
#x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.2, random_state=0)
        
# lgb

#for n_fold, (trn_idx, val_idx) in enumerate(folds.split(f_train)):
#    trn_x, trn_y = f_train.ix[trn_idx], target[trn_idx]
#    val_x, val_y = f_train.ix[val_idx], target[val_idx]
#    d_train = lgb.Dataset(trn_x, label=trn_y)
#    d_valid = lgb.Dataset(val_x, label=val_y)
#    watchlist = [d_train, d_valid]
#    model = lgb.train(lgb_params, train_set=d_train, num_boost_round=240, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=10) 
#    fold = model.predict(val_x[train_features])
#    print (log_loss(val_y, fold))
#    preds = preds + (np.array(model.predict(lgb.Dataset(f_test[train_features]))) / (5*1))

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(f_train)):
    trn_x, trn_y = f_train.ix[trn_idx], target[trn_idx]
    val_x, val_y = f_train.ix[val_idx], target[val_idx]


    d_train = lgb.Dataset(trn_x, label=trn_y)
    d_valid = lgb.Dataset(val_x, label=val_y)

    watchlist = [d_train, d_valid]
    iterations = 4

    for i in range(0, iterations):

        model = lgb.train(lgb_params, train_set=d_train, num_boost_round=240, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=10) 
        fold = model.predict(val_x[train_features])
        df_aux = pd.DataFrame()
        df_aux['sid'] = sid
        preds = preds + (np.array(model.predict(f_test[train_features])) / (5*1))
        df_aux['preds'] = preds
        oof_preds = pd.concat([oof_preds, df_aux])
        #result += log_loss(val_y, fold)/ (5*iterations)

oof_preds.to_csv("lgb_001.csv", index=False)

print("FOLDS RESULT: ", result)

f_test['sid'] = sid
f_test['target'] = preds

submission = f_test[['sid','target']]
submission.to_csv("lgb_test_001.csv", index=False)





