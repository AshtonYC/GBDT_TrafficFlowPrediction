### GDBT
# Load
import numpy as np
from regression_model.GBDT import GBDTRegressor
import pickle
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

Norm = 1e6
# Load training and testing dataset
train_data = pd.read_csv('../Train/Train_Max_Last.csv')
train_data = shuffle(train_data)
Y_data = train_data['VMT (Veh-Miles)']
X_data = train_data.drop('VMT (Veh-Miles)', axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X_data, Y_data, test_size=0.3)
# df_X = pd.DataFrame(X_data)
X_train = X_train.values
X_valid = X_valid.values
# df_y = pd.DataFrame(Y_data)
y_train_mid = y_train.values / Norm
y_train = y_train_mid.reshape(len(y_train_mid), )
y_valid_mid = y_valid.values / Norm
y_valid = y_valid_mid.reshape(len(y_valid_mid), )
test_data = pd.read_csv('../Test/New_Test_0502_Last.csv')

Y_test_data = test_data['VMT (Veh-Miles)']
X_test_data = test_data.drop('VMT (Veh-Miles)', axis=1)

df_X_test = pd.DataFrame(X_test_data)
X_test = df_X_test.values
df_y_test = pd.DataFrame(Y_test_data)
y_test_mid = df_y_test.values / Norm
y_test = y_test_mid.reshape(len(y_test_mid), )

# Build GBDT model
# model = GBDTRegressor()
#
# # Model Train
# model.fit(X_train, y_train, X_valid, y_valid)
#
# # Save model
# pickle.dump(model, open("pima.pickle_test_Last3.dat", "wb"))

# Load model
loaded_model = pickle.load(open("pima.pickle_test_Last2.dat", "rb"))

# Prediction from loaded model
y_pred = loaded_model.predict(X_test) * Norm
predictions = np.array([round(value) for value in y_pred])

# evaluate predictions
a = predictions.tolist()
y_true = np.array([round(value * Norm) for value in y_test])
b = y_true.tolist()

Error = mean_absolute_percentage_error(a, b)
count = 0
E = np.zeros_like(predictions)
for i in range(len(predictions)):
    E[i] = (predictions[i] - y_true[i])
    if E[i] > 0.2:
        count += 1
E_20 = count / len(y_true)
print("mean_absolute_percentage_error: %.2f%%" % (Error * 100.0))
print("accuracy of 20%%+: %0.2f%%" % (E_20 * 100.0))
y_result = np.vstack((y_pred, y_test))
result_data = pd.DataFrame(y_result)

Hour = np.linspace(0, y_result.shape[1], y_result.shape[1])
plt.plot(Hour, y_pred, Hour, y_true, ls="-", lw=2, label="plot figure")

plt.legend(['Prediction', 'Actual'])

plt.show()
