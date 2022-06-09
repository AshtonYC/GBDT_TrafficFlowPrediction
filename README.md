# GBDT_TrafficFlowPrediction
## Introduction
This project is aiming to develop a GBDT algorithm that could predict the short-term traffic flow values accroding to the pervious traffic flow data. The example training, validation, and testing data are obtained from the PeMS, and it is for the Freeway SR52-E in District 11 (San Diego). And the feature of the data concluding pervious VMT, VHT, Month, Day, Hour, number of lane point, and the precentage of observed car. The Gradient Boosting Decision Tree (GBDT), a popular machine learning algorithm in classification and regression field and needs to scan all the data instances to estimate the information gain of all possible split points, is a well-developed algorithm. In this case, the SR52-E connects the University City to the other area, so the prediction of VMT could be used to estimate the rough lane occupancy, which might provide the information to help to schedule the maintenance plan. 

## Testing example
Run the GBDT_reg_test.py to test the model. If you want to test the existing model, comment out the following lines in GBDT_reg_test.py would be fine. 
*model = GBDTRegressor()
*model.fit(X_train, y_train, X_valid, y_valid)
*pickle.dump(model, open("pima.pickle_test_Last3.dat", "wb"))

## Data split
The training data and validation data are from 2021, and they are are randomly scrambled and 70% of them are assigned as training data, the rest 30% are the validation data.

## Tunable parameter
The accuracy of the GBDT could be changed by tuning the number of CART esimators, the learning rate, and some stopping criterias in the GBDT.py file. The loss function and the corresponding gradient can be changed in the GBDT.py file.
