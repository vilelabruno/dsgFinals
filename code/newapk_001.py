import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import xgboost as xgb

path = '~/dsg_paris/'

def transform_time(df_train):
	df_train['time'], df_train['mili'] = df_train['duration'].str.split('.', 1).str
	df_train['days'], df_train['time'] = df_train['time'].str.split(' days ', 1).str
	df_train['hour'], df_train['min'], df_train['sec'] = df_train['time'].str.split(':', 2).str
	df_train['seconds'] = df_train['sec'].astype(int) + (df_train['min'].astype(int)*60) + (df_train['hour'].astype(int)*3600) + (df_train['days'].astype(int)*86400)
	df_train = df_train.drop(columns = ['days', 'hour', 'min', 'sec', 'mili', 'time'])
	return df_train

print("Loading files...")
train = pd.read_csv("train_tracking.csv")
test  = pd.read_csv("test_tracking.csv")

target = pd.read_csv("train_session.csv")
products = pd.read_csv("dsg18_cdiscount_sku_categ.csv")
products.columns = ['sku','categ_1','categ_2','categ_3']

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

purchase = ['num_PURCHASE_PRODUCT_CAROUSEL', 'num_PURCHASE_PRODUCT_LP','num_PURCHASE_PRODUCT_LR', 'num_PURCHASE_PRODUCT_PA','num_PURCHASE_PRODUCT_SHOW_CASE', 'num_PURCHASE_PRODUCT_UNKNOW_ORIGIN']

f_train['num_PURCHASES'] = (f_train['num_PURCHASE_PRODUCT_CAROUSEL']+f_train['num_PURCHASE_PRODUCT_LP']
	+f_train['num_PURCHASE_PRODUCT_LR']+f_train['num_PURCHASE_PRODUCT_PA']+f_train['num_PURCHASE_PRODUCT_SHOW_CASE']+f_train['num_PURCHASE_PRODUCT_UNKNOW_ORIGIN'])

f_test['num_PURCHASES'] = (f_test['num_PURCHASE_PRODUCT_CAROUSEL']+f_test['num_PURCHASE_PRODUCT_LP']
	+f_test['num_PURCHASE_PRODUCT_LR']+f_test['num_PURCHASE_PRODUCT_PA']+f_test['num_PURCHASE_PRODUCT_SHOW_CASE']+f_test['num_PURCHASE_PRODUCT_UNKNOW_ORIGIN'])

f_train['Count_Div_Purchases'] = f_train['session_count'] / f_train['num_PURCHASES']
f_test['Count_Div_Purchases'] = f_test['session_count'] / f_test['num_PURCHASES']

# Percentage of each type
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
merge = pd.merge(merge, products, on='sku', how='left', sort=False)

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
cat_columns = ['query','sku','sname','oquery','offerid','stype','facets','idcar','siteid','categ_1','categ_2','categ_3']
rst_columns = []
for col in cat_columns:
	encoding = merge.groupby(col)['type'].count().reset_index()
	encoding.columns = [col, col+'_frequency']
	rst_columns.append(col+"_frequency")
	merge = pd.merge(merge, encoding, on=col, how='left', sort=False)

nrm_columns = []
for col in rst_columns:
	encoding = merge.groupby('sid')[col].max().reset_index()
	encoding.columns = ['sid',col+'_max']
	nrm_columns.append(col+'_max')
	f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
	f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in rst_columns:
	encoding = merge.groupby('sid')[col].min().reset_index()
	encoding.columns = ['sid',col+'_min']
	nrm_columns.append(col+'_min')
	f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
	f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in rst_columns:
	encoding = merge.groupby('sid')[col].mean().reset_index()
	encoding.columns = ['sid',col+'_mean']
	nrm_columns.append(col+'_mean')
	f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
	f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in rst_columns:
	encoding = merge.groupby('sid')[col].std().reset_index()
	encoding.columns = ['sid',col+'_std']
	nrm_columns.append(col+'_std')
	f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
	f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)
for col in rst_columns:
	encoding = merge.groupby('sid')[col].sum().reset_index()
	encoding.columns = ['sid',col+'_sum']
	nrm_columns.append(col+'_sum')
	f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
	f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)

# lvl 2 encoding
for col in cat_columns:
	encoding = merge.groupby(['sid',col])['type'].count().reset_index()
	encoding.columns = ['sid', col, col+'_sid_frequency']
	lvl2_encoding = encoding.groupby('sid')[col].count().reset_index()
	lvl2_encoding.columns = ['sid',col+'_count_sid']
	f_train = pd.merge(f_train, lvl2_encoding, on='sid', how='left', sort=False)
	f_test  = pd.merge(f_test, lvl2_encoding, on='sid', how='left', sort=False)

# Target Encoding with categories
f_train = pd.merge(f_train, target, on='sid', how='left', sort=False)
ctg_columns = ['categ_1_frequency_max','categ_2_frequency_max','categ_3_frequency_max','categ_1_frequency_min','categ_2_frequency_min','categ_3_frequency_min']
for col in ctg_columns:
	encoding = f_train.groupby(col)['target'].mean().reset_index()
	encoding.columns = [col, 'encoding_'+col]
	f_train = pd.merge(f_train, encoding, on=col, how='left', sort=False)
	f_test  = pd.merge(f_test, encoding, on=col, how='left', sort=False)

merge['ff'][merge['ff'] == True]  = 1
merge['ff'][merge['ff'] == False] = 0
encoding = merge.groupby('sid')['ff'].sum().reset_index()
encoding.columns = ['sid','ff_sum']
f_train = pd.merge(f_train, encoding, on='sid', how='left', sort=False)
f_test  = pd.merge(f_test, encoding, on='sid', how='left', sort=False)

f_train.to_csv(path+'input/train_xgb.csv', index=False)
f_test.to_csv(path+'input/test_xgb.csv', index=False)


target = f_train['target']
sid = f_test['sid']
f_train.drop(["sid", "target"], axis=1, inplace=True)
f_test.drop("sid", axis=1, inplace=True)
print(f_train.columns)

# Training Session
print("Starting Kfolds...")
folds = KFold(n_splits=5, shuffle=True, random_state=159)

preds = np.repeat(0, len(f_test))

result = 0

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(f_train)):
    trn_x, trn_y = f_train.ix[trn_idx], target[trn_idx]
    val_x, val_y = f_train.ix[val_idx], target[val_idx]
    d_train = xgb.DMatrix(trn_x, trn_y)
    d_valid = xgb.DMatrix(val_x, val_y)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    iterations = 1
    for i in range(0, iterations):
	    params = {
	        'eta': 0.01,
	        'max_depth': 5, 
	        'subsample': 0.9, 
	        'colsample_bytree': 0.9, 
	        'colsample_bylevel':0.9,
	        #'min_child_weight':10,
	        #'alpha':4,
	        'objective': 'binary:logistic',
	        'eval_metric': 'logloss', 
	        'nthread':64,
	        'seed': 3*i, 
	        'silent': True}
	    model = xgb.train(params, d_train, 10000, watchlist, 
	                  maximize=False, early_stopping_rounds = 50, 
	                  verbose_eval=100)
	    fold = model.predict(xgb.DMatrix(val_x))
	    preds = preds + (np.array(model.predict(xgb.DMatrix(f_test))) / (5*iterations))
	    result += log_loss(val_y, fold)/ (5*iterations)

print("FOLDS RESULT: ", result)

f_test['sid'] = sid
f_test['target'] = preds

submission = f_test[['sid','target']]
submission.to_csv(path+"stacking/newapk_001.csv", index=False)