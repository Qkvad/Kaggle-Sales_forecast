import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler

df_features = pd.read_csv('Features data set.csv')
pd.DataFrame(df_features).to_csv('Features_data_set.csv',
                              sep=';',
                              encoding='cp1250',
                              decimal=',')
pd.set_option('display.max_columns', None)
print(df_features.head(10))
df_features_column_headers = df_features.columns.values.tolist()
print(df_features_column_headers)
print(df_features.isnull().sum(axis = 0))
print(df_features.describe())
df_features['CPI'] = df_features['CPI'].fillna(df_features['CPI'].median())
df_features['Unemployment'] = df_features['Unemployment'].fillna(df_features['Unemployment'].median())
df_features = df_features.fillna(0)
print(df_features.isnull().sum(axis = 0))
print(df_features.describe())

df_stores = pd.read_csv('stores data-set.csv')
pd.DataFrame(df_stores).to_csv('stores_data_set.csv',
                              sep=';',
                              encoding='cp1250',
                              decimal=',')
print(df_stores.describe())
print(df_stores.isnull().sum(axis = 0))

df_sales = pd.read_csv('sales data-set.csv')
pd.DataFrame(df_sales).to_csv('sales_data_set.csv',
                              sep=';',
                              encoding='cp1250',
                              decimal=',')
print(df_sales.describe())
print(df_sales.isnull().sum(axis = 0))

df_sales_features = pd.merge(df_features,
                             df_sales,
                             on=['Store', 'Date', 'IsHoliday'],
                             how='right')
print(df_sales_features.head(10))
df_features_column_headers_sales_features = df_sales_features.columns.values.tolist()
print(df_features_column_headers_sales_features)
print(df_sales_features.shape)
print(df_sales_features.isnull().sum(axis = 0))
df = pd.merge(df_sales_features,
              df_stores,
              on=['Store'],
              how='left')
df_headers = df.columns.values.tolist()
print(df_headers)
print(df.shape)

df_ws = df[['Weekly_Sales']]
df=df.drop(columns=['Weekly_Sales'])
df['Weekly_Sales'] = df_ws

#df = pd.to_datetime(df['Date']) #----------------------

df_headers = df.columns.values.tolist()
print(df_headers)
print(df.shape)

dff = df
datess = dff.Date.unique()
i = 0
while i < len(datess):
    dff.loc[dff.Date == datess[i], 'Date'] = i
    i += 1
dates = dff.Date.unique()
pd.DataFrame(dff).to_csv('dff.csv',
                              sep=';',
                              encoding='cp1250',
                              decimal=',')
df = df.drop(columns=['Dept', 'Type', 'Size'])

dates = df.Date.unique()
print(len(dates))
i = 0
while i < len(dates):
    df.loc[df.Date == dates[i], 'Date'] = i
    i += 1
dates = df.Date.unique()
print(dates)



df = df.drop(columns=['Store'])
df_all_stores_accumulated = df.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
df_all_stores_accumulated = df_all_stores_accumulated.sort_values('Date', ascending=True)
print(df_all_stores_accumulated)

dff = df[['Date', 'Temperature']]
df_temp = dff.groupby(by=['Date'], as_index=False)['Temperature'].mean()
df_temp = df_temp.sort_values('Date', ascending=True)
df_temp = df_temp.drop(columns=['Date'])
print(df_temp)
dff = df[['Date', 'CPI']]
df_CPI = dff.groupby(by=['Date'], as_index=False)['CPI'].mean()
df_CPI = df_CPI.sort_values('Date', ascending=True)
df_CPI = df_CPI.drop(columns=['Date'])
dff = df[['Date', 'Fuel_Price']]
df_Fuel_Price = dff.groupby(by=['Date'], as_index=False)['Fuel_Price'].mean()
df_Fuel_Price = df_Fuel_Price.sort_values('Date', ascending=True)
df_Fuel_Price = df_Fuel_Price.drop(columns=['Date'])
dff = df[['Date', 'IsHoliday']]
df_isHoliday = dff.groupby(by=['Date'], as_index=False)['IsHoliday'].median()
df_isHoliday = df_isHoliday.sort_values('Date', ascending=True)
df_isHoliday = df_isHoliday.drop(columns=['Date'])
dff = df[['Date', 'Unemployment']]
df_Unemployment = dff.groupby(by=['Date'], as_index=False)['Unemployment'].mean()
df_Unemployment = df_Unemployment.sort_values('Date', ascending=True)
df_Unemployment = df_Unemployment.drop(columns=['Date'])

dff = df[['Date', 'MarkDown1']]
df_MD1 = dff.groupby(by=['Date'], as_index=False)['MarkDown1'].mean()
df_MD1 = df_MD1.sort_values('Date', ascending=True)
df_MD1 = df_MD1.drop(columns=['Date'])
dff = df[['Date', 'MarkDown2']]
df_MD2 = dff.groupby(by=['Date'], as_index=False)['MarkDown2'].mean()
df_MD2 = df_MD2.sort_values('Date', ascending=True)
df_MD2 = df_MD2.drop(columns=['Date'])
dff = df[['Date', 'MarkDown3']]
df_MD3 = dff.groupby(by=['Date'], as_index=False)['MarkDown3'].mean()
df_MD3 = df_MD3.sort_values('Date', ascending=True)
df_MD3 = df_MD3.drop(columns=['Date'])
dff = df[['Date', 'MarkDown4']]
df_MD4 = dff.groupby(by=['Date'], as_index=False)['MarkDown4'].mean()
df_MD4 = df_MD4.sort_values('Date', ascending=True)
df_MD4 = df_MD4.drop(columns=['Date'])
dff = df[['Date', 'MarkDown5']]
df_MD5 = dff.groupby(by=['Date'], as_index=False)['MarkDown5'].mean()
df_MD5 = df_MD5.sort_values('Date', ascending=True)
df_MD5 = df_MD5.drop(columns=['Date'])

df_all_stores_accumulated['Temperature'] = df_temp
df_all_stores_accumulated['CPI'] = df_CPI
df_all_stores_accumulated['Fuel_Price'] = df_Fuel_Price
df_all_stores_accumulated['isHoliday'] = df_isHoliday
df_all_stores_accumulated['Unemployment'] = df_Unemployment

df_all_stores_accumulated['MarkDown1'] = df_MD1
df_all_stores_accumulated['MarkDown2'] = df_MD2
df_all_stores_accumulated['MarkDown3'] = df_MD3
df_all_stores_accumulated['MarkDown4'] = df_MD4
df_all_stores_accumulated['MarkDown5'] = df_MD5

df_ws = df[['Weekly_Sales']]
df_all_stores_accumulated=df_all_stores_accumulated.drop(columns=['Weekly_Sales'])
df_all_stores_accumulated['Weekly_Sales'] = df_ws

print(df_all_stores_accumulated)

df_final = df_all_stores_accumulated.drop(columns=['Date'])
labels = df_final.columns.values.tolist()
data = df_final.values
X, Y = data[:, :-1], data[:, -1]
#scaler = StandardScaler()
#X = scaler.fit_transform(x)
X_train = X[:-7]
X_test = X[-8:]
Y_train = Y[:-7]
Y_test = Y[-8:]

#-------------------------------------------------------------------------
#                                   XGB
#-------------------------------------------------------------------------
xgb_r = xgb.XGBRegressor(objective='reg:linear',
                        n_estimators=10, seed=123)
xgb_r.fit(X_train, Y_train)
pred = xgb_r.predict(X_test)
xgb_mse = MSE(Y_test, pred)

xgb_rmse = np.sqrt(MSE(Y_test, pred))
print("MSE : % f" % (xgb_mse))
print("RMSE : % f" % (xgb_rmse))

t1 = (abs(Y_test - pred)).mean()
true_mean = Y_test.mean()
t2 = true_mean - t1
xgb_score = t2/true_mean
print("XGB score: ", xgb_score)

plt.plot(dates[:-7], Y_train, c='b')
plt.plot(dates[:-7], Y_train, c='r')
plt.plot(dates[-8:], Y_test, c='b', label='test')
plt.plot(dates[-8:], pred, c='r', label='prediction')
plt.legend()
plt.title('XGBoost prediction')
plt.show()

xgb_importances = np.ndarray(len(labels)-1)
importance = xgb_r.feature_importances_
for i,v in enumerate(importance):
   xgb_importances[i] = v
   print('Score: %.5f' % v, '  Feature: ', labels[i])

plt.bar(labels[:-1], xgb_importances, color='red', width=0.2)
plt.title('XGBoost feature importance')
plt.show()

#-------------------------------------------------------------------------
#                             Linear Regression
#-------------------------------------------------------------------------
regr = LinearRegression()

regr.fit(X_train, Y_train)
y_pred = regr.predict(X_test)
print("Linear regression score: ", regr.score(X_test, Y_test))

plt.plot(dates[:-7], Y_train, c='b')
plt.plot(dates[:-7], Y_train, c='r')
plt.plot(dates[-8:], Y_test, c='b', label='test')
plt.plot(dates[-8:], y_pred, c='r', label='prediction')
plt.legend()
plt.title('Linear regression prediction')
plt.show()

mae = mean_absolute_error(y_true=Y_test, y_pred=y_pred)
mse = mean_squared_error(y_true=Y_test, y_pred=y_pred)
rmse = mean_squared_error(y_true=Y_test, y_pred=y_pred, squared=False)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

t1 = (abs(Y_test - y_pred)).mean()
true_mean = Y_test.mean()
t2 = true_mean - t1
lr_score = t2/true_mean
print("LR score: ", lr_score)







