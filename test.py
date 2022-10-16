from auroc_comparsion import *

response = np.array([0,0,0,0,0,0,1,1,1,1,1,1,1]) # GT
model1_prediction = np.array([0.1,0.2,0.05,0.3,0.1,0.6,0.6,0.7,0.8,0.99,0.8,0.67,0.5]) # prediction of model1
model2_prediction = np.array([0.3,0.6,0.2,0.1,0.1,0.9,0.23,0.7,0.9,0.4,0.77,0.3,0.89]) # prediction of model2

# return auroc of each model and p-value of auroc comparison
aucs, p_value = delong_roc_test(response, model1_prediction, model2_prediction)

print(f' aucs : {aucs}, p_value : {p_value}') 
>>> aucs : [0.96428571 0.73809524], p_value : 0.09452572880558979
