#%% Importações

import numpy as np
import pandas as pd
import xgboost as xgb

#%% Lendo arquivos

path = "~/ga_revenue/in/"
path2 = "~/ga_revenue/out/"

test_df = pd.read_csv(path+"test-flattened.csv")
train_df = pd.read_csv(path+"train-flattened.csv")

test_df = test_df.groupby(['fullVisitorId'])

for i in test_df:
	print(i + ':')
	print(test_df[i].head(5))

#%% Dropando colunas por análise

train_df = train_df.drop('sessionId', axis=1)


test_df = test_df.drop('sessionId', axis=1)

#%% Retirando o target

y_train = train_df['totals.transactionRevenue'].fillna(0).map(lambda x: np.log(x) if (x > 0) else 0)
train_df = train_df.drop('totals.transactionRevenue', axis=1)

#%% Mostrando a tabela

columns = train_df.columns

for x in columns:
	labels, uniques = pd.factorize(train_df[x])
	if(uniques.size == 1):
		train_df = train_df.drop(x, axis=1)
	elif(uniques.size <= 1):
		ohe = pd.get_dummies(train_df[x])
		colohe = ohe.columns
		for j in colohe:
			train_df[str(x) + '.' + str(j)] = ohe[j]
		train_df = train_df.drop(x, axis=1)
	else:
		train_df[x] = labels

columns = test_df.columns

for x in columns:
	labels, uniques = pd.factorize(test_df[x])
	if(uniques.size == 1):
		test_df = test_df.drop(x, axis=1)
	elif(uniques.size <= 10):
		ohe = pd.get_dummies(test_df[x])
		colohe = ohe.columns
		for j in colohe:
			test_df[str(x) + '.' + str(j)] = ohe[j]
		test_df = test_df.drop(x, axis=1)
	else:
		test_df[x] = labels

train_columns = train_df.columns
test_columns = test_df.columns

train_col = train_columns.difference(test_columns)
for i in train_col:
	train_df = train_df.drop(i, axis=1)

test_col = test_columns.difference(train_columns)
for i in test_col:
	test_df = test_df.drop(i, axis=1)