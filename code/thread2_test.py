import pandas as pd

pathin = "~/input/"
test = pd.read_csv(pathin + 'test_tracking.csv')
aux = pd.DataFrame()
for i in range(0, 909):
	aux = test.loc[i*1000:((i+1)*1000)-1]
	aux.to_csv('data'+ str(i) +'.csv', index=False)