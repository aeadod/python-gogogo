import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,accuracy_score


df = pd.read_csv('C:/Users/aeadod/Desktop/project/exercise/e8/data/bank-additional-full.csv',sep=',')
print(df.head())
print('----------------------------------------------------------------------------------------')
x_label = LabelEncoder()
df['job']=x_label.fit_transform(df['job'])
df['marital']=x_label.fit_transform(df['marital'])
df['education']=x_label.fit_transform(df['education'])
df['default']=x_label.fit_transform(df['default'])
df['housing']=x_label.fit_transform(df['housing'])
df['loan']=x_label.fit_transform(df['loan'])
df['contact']=x_label.fit_transform(df['contact'])
df['month']=x_label.fit_transform(df['month'])
df['day_of_week']=x_label.fit_transform(df['day_of_week'])
df['poutcome']=x_label.fit_transform(df['poutcome'])
df['y']=x_label.fit_transform(df['y'])
print(df.head())
print('----------------------------------------------------------------------------------------')
y=df['y']
df.drop(['y'],inplace=True,axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=42)
clf1=DecisionTreeClassifier(criterion='entropy')
clf2=KNeighborsClassifier()
#clf=RandomForestClassifier()
clf1.fit(x_train,y_train)
clf2.fit(x_train,y_train)

# y_pred1=clf1.predict(x_test)
# y_pred2=clf2.predict(x_test)

# p1 = precision_score(y_test, y_pred1, average='binary')
# p2= precision_score(y_test, y_pred2, average='binary')
# r1 = recall_score(y_test, y_pred1, average='binary')
# r2= recall_score(y_test, y_pred2, average='binary')
# score1 = accuracy_score(y_test, y_pred1)
# score2=accuracy_score(y_test, y_pred2)
# print('-------------------------------------------------------------------')
# print('决策树准确率{:.2f}'.format(p1))
# print('KNN准确率',p2)
# print('决策树召回率',r1)
# print('KNN召回率',r2)
# print('ACC OF 决策树',score1)
# print('ACC OF KNN',score2)

y_pred1=clf1.predict(x_train)
y_pred2=clf2.predict(x_train)

p1 = precision_score(y_train, y_pred1, average='binary')
p2= precision_score(y_train, y_pred2, average='binary')
r1 = recall_score(y_train, y_pred1, average='binary')
r2= recall_score(y_train, y_pred2, average='binary')
score1 = accuracy_score(y_train, y_pred1)
score2=accuracy_score(y_train, y_pred2)
print('-------------------------------------------------------------------')
print('决策树准确率{:.2f}'.format(p1))
print('KNN准确率',p2)
print('决策树召回率',r1)
print('KNN召回率',r2)
print('ACC OF 决策树',score1)
print('ACC OF KNN',score2)