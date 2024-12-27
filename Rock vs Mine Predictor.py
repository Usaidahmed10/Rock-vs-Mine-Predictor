'''
Steps:
1. Data Loading
2. Data Preprocessing
3. Splitting data into train and test sets
4. Applying Logistic Regression model (since this is a binary classification problem)
5. Training the Logistic Regression model
6. Model Evaluation
7. Predictive System
'''

import numpy as np # for mnaths
import pandas as pd # for data processin
from sklearn.model_selection import train_test_split  # to split into train and test data
from sklearn.linear_model import LogisticRegression  # to train model with logistic regression
from sklearn.metrics import accuracy_score  # to calculate accuracy of our model 
from sklearn.preprocessing import StandardScaler  # for feature scaling

# Data Loading  
try:
    sonar_data = pd.read_csv('Copy of sonar data.csv', header=None)
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()


# Data Processing
print(sonar_data.head())   # getting an idea of the data 
print(sonar_data.describe())   # statistical nmeasure of the data
print(sonar_data[60].value_counts())   # checking how many total mines and rocks in the data (M->Mine, R->Rock)
print(sonar_data.groupby(60).mean())   # mean for each column value, grouped my R and M

# Separating Data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.1,   # specifies that 10% of the data will be used as the test set
    stratify=Y,   # ensures the distribution of classes in the target variable (Y) is preserved between the train and test sets (equal ratio of R and M in both sets)
    random_state=1   # sets a fixed seed for reproducibility so that the same split is generated every time the code is run.
)
print(X.shape, X_train.shape, X_test.shape)   #seeing the split data sizes

# Feature Scaling
# scaling features ensures that all input features are on the same scale (mean = 0, std = 1)
scaler = StandardScaler()

# fitting the scaler on training data and transforming both training and test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = model.predict(X_train)   #accuracy on training data
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 
print(f'Accuracy on training data: {training_data_accuracy:.2%}')
X_test_prediction = model.predict(X_test)   #accuracy on test data
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)   
print(f'Accuracy on test data: {test_data_accuracy:.2%}')

# Predictive System 
# input data can be any values that we come across
input_data = (0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# scale the input data using the same scaler fitted on training data
input_data_scaled = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_scaled)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')