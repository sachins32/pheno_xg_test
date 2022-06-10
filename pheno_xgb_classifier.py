import pandas as pd
import pickle
import daal4py as d4p

class pheno_classifier():
    
    def __init__(self):
        
        path = "dall4py_classifier(1).pkl"
        with open("dall4py_classifier(1).pkl", "rb") as model_file:
            pickle.load(model_file)
            
        # self.clf = clf
        
    def predict(self, X):
        # Make predictions
        daal_prediction = d4p.gbt_regression_prediction().compute(X, self.clf)
        return daal_prediction

# X_test = pd.read_csv("X_test.csv")
# print("Shape of df :{}".format(X_test.shape))

# classifier = pheno_classifier()
# y_prob = classifier.predict(X_test)

# print("Prediction Probability are : {}".format(y_prob))