import pandas as pd
import numpy as np

#读取数据
data = pd.read_csv("data/credit card.csv")
#print(data.head())


#查看空值
#print(data.isnull().sum())

#查看交易类型 type
#print(data.type.value_counts())

#图显
'''
type = data["type"].value_counts()
transactions = type.index
quantity = type.values


import plotly.express as px
figure = px.pie(type, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()
'''

# Checking correlation通过计算相关系数来分析两个或多个变量之间是否存在关联性
#data.corr 新版本用法无法自动识别string，添加numeric_only=True
correlation = data.corr(numeric_only=True)
#print(correlation["isFraud"].sort_values(ascending=False))


#替换

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
#print(data.head())

# 拆分数据
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

# 训练机器学习模型   决策树回归器
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# prediction 进行预测
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))



