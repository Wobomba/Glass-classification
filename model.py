import numpy as np
import pandas as pd

#deployinf the model, start with importing joblib
import joblib

dataset = pd.read_csv('glass.csv')
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,9]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=42)
cls.fit(X_train, y_train)

#y_pred = cls.predict(X_test)

print('ACCURACY is', cls.score(X_test, y_test)*100, '%')

#then save the model which is the 2nd step
filename = 'finalized_model.sav'
joblib.dump(cls, filename)
