import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM

df= pd.read_csv("C:\\Users\\aeadod\\Desktop\\project\\exercise\\e12\\data\\STOCK.csv",sep=',')
# plt.figure(figsize=(16,8))
# plt.plot(df['Close'].values[::-1],label='Close Price History')
# plt.show()

zb=df['Close'].values[::-1]

scaler=MinMaxScaler()
all_data=scaler.fit_transform(np.array(zb).reshape(-1,1))
test_data=all_data[987:]

x_train,y_train=[],[]
for i in range(10,987):
    x_train.append(all_data[i-10:i])
    y_train.append(all_data[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

model=Sequential()
model.add(LSTM(units=32,input_shape=(x_train.shape[1],1)))#一层lstm
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train)

print(len(test_data))
x_vaild=[]
for i in range(10,len(test_data)):
    x_vaild.append(test_data[i-10:i])
x_vaild=np.array(x_vaild)
print(x_vaild.shape)
closing_price=model.predict(x_vaild)
closing_price=scaler.inverse_transform(closing_price)


plt.figure(figsize=(16,8))
a=df['Close'].values[::-1]
plt.plot(a[987:],label='Close Price History')
#plt.plot(closing_price,label='pre')
plt.show()
