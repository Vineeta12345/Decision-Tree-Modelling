# -*- coding: utf-8 -*-
"""
Created on Wed May  6 01:15:10 2020

@author: Vineeta
"""

import pandas as pd
df = pd.read_csv("DecisionTree.csv")
df.head()
inputs = df.drop('DEAFULTED',axis='columns')
target = df['DEAFULTED']
inputs
target
from sklearn.preprocessing import LabelEncoder
le_age =LabelEncoder()
le_home =LabelEncoder()
le_income =LabelEncoder()
le_gender =LabelEncoder()
le_household_n =LabelEncoder()
le_credit_lines_n =LabelEncoder()
inputs['age_n'] = le_age.fit_transform(inputs['AGE'])
inputs['home_n'] = le_home.fit_transform(inputs['HOME'])
inputs['income_n'] = le_income.fit_transform(inputs['INCOME'])
inputs['gender_n'] = le_gender.fit_transform(inputs['GENDER'])
inputs['house_hold_n'] = le_household_n.fit_transform(inputs['HOUSEHOLD_N'])
inputs['cred_lines_n'] = le_credit_lines_n.fit_transform(inputs['CREDIT_LINES_N'])
inputs.head()
inputs_n = inputs.drop(['AGE','HOME','INCOME','GENDER','HOUSEHOLD_N',
                        'CREDIT_LINES_N'],axis='columns')
inputs_n
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n,target)
model_new = model.fit(inputs_n,target)
model.score(inputs_n,target)
model.predict([[29,1,0,0,2,2]])
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import pydotplus
feature_names=["age_n", "home_n", "income_n", "gender_n", "house_hold_n", "cred_lines_n"]
class_names=["1", "0"]
dot_data = tree.export_graphviz(model_new, out_file=None,
                                feature_names=feature_names,
                                class_names=class_names)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("decisiontree.png")
graph.write_pdf("decisiontree.pdf")
























