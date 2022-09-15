# import files
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

# data gathering
df = pd.read_csv('diabetes.csv')

#Select target
x = df.drop('Outcome', axis =1)
y = df['Outcome']

# data scaling
std_scaler = StandardScaler()
std_scaler.fit(x)
std_array = std_scaler.transform(x)
x_std = pd.DataFrame(std_array, columns=x.columns)

# data spliting
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=24, stratify=y)

# model building
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)

# hyper parameter tunning
knn_clf = KNeighborsClassifier()
hyp = {'n_neighbors': np.arange(2,30),
      'p':[1,2]}

gscv_knn = GridSearchCV(knn_clf, hyp, cv = 5)
gscv_knn.fit(x_train, y_train)

#pickle pile
pickle.dump(gscv_knn, open('diabetes.pkl', 'wb'))