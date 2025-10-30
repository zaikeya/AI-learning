import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/advertising.csv")
#print(data.head())
#print(data.isnull().sum())

import plotly.express as px
import plotly.graph_objects as go

'''
figure = px.scatter(data_frame = data, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()
'''
figure = px.scatter(data_frame = data, x="Sales",
                    y="Newspaper", size="Newspaper", trendline="ols")
figure.show()


correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))


#data.drop  作用是删除指定列或者行，默认删除行，需要加axis=1 表示列,不要省略axis=
x = np.array(data.drop(['Sales'],  axis=1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


#线性回归
model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

#features = [[TV, Radio, Newspaper]]
features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))

