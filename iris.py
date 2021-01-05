# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:59:53 2021

@author: Yashasvee Shukla
"""

import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('E:\Practice\Iris Plant Prediction\iris.data')

X = np.array(df.iloc[:, 0:4]) 
y = np.array(df.iloc[:, 4:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)



pickle.dump(sv, open('iris.pk1', 'wb'))