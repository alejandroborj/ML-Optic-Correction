#%%
import matplotlib.pyplot as plt

import tfs
import pandas as pd
import numpy as np


errors = [-8,-6,-4,-2, 0, 2, 4, 6, 8]
#errors = [-0.1, -0.2, -0.3, -0.4, -0.5, -1.4, -3, 1.5, 1, 2.5]
rms_rdt = []

rdt="RE_103000"
rdt="RE_022000"

for error in errors:
    rdt_data = pd.read_csv(f"./datasets/tests/rdt_responses_2/samples/MQXA.1R1_0.0_0.0_0.0_{error*1e-4}_b1.csv", sep="\t")
    
    #rms_rdt.append(np.mean(np.real(rdt_data[rdt])))
    #print(error, rdt_data["NAME"][15])
    bpm_name = rdt_data["NAME"][15]
    rms_rdt.append(rdt_data[rdt][15])
    #rms_rdt.append(np.abs(rdt_data[rdt][500]))


#plt.yscale('log', base=2)

plt.title("")
plt.xlabel(fr"{bpm_name}: {rdt}")
plt.ylabel(r"MQXA.1R1 $a_4^2$ Error Squared")
#plt.plot(np.array(errors), rms_rdt)
plt.plot(rms_rdt, np.array(errors))

print(rms_rdt)
print(errors)

plt.show()


plt.clf()
plt.xlabel(fr"{bpm_name}: sqrt(|{rdt}|)")
plt.ylabel(r"MQXA.1R1 $a_4$ Error Squared")
#plt.plot(np.array(errors), rms_rdt)
plt.plot((abs(np.array(rms_rdt)))**0.5, np.array(errors))
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#define predictor and response variables
x = np.array([83767.7592, 70109.8361, 28165.6793, -39842.4639, -130397.8359])
y = np.array([0, 2, 4, 6, 8])



x_poly = np.linspace(min(x), max(x), 100)


#specify degree of 3 for polynomial regression model
#include bias=False means don't force y-intercept to equal zero
poly = PolynomialFeatures(degree=2, include_bias=False)

#reshape data to work properly with sklearn
poly_features = poly.fit_transform(x.reshape(-1, 1))

poly_features_poly = poly.fit_transform(x_poly.reshape(-1, 1))


#fit polynomial regression model
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)

#display model coefficients
print(poly_reg_model.intercept_, poly_reg_model.coef_)

#use model to make predictions on response variable
y_predicted = poly_reg_model.predict(poly_features)

y_predicted_poly = poly_reg_model.predict(poly_features_poly)

plt.figure(figsize=(1.3*4.2, 1.3*2.5))
#create scatterplot of x vs. y
plt.scatter(x, y, label="Data", color="tab:orange")
#add line to show fitted polynomial regression model
plt.plot(x_poly, y_predicted_poly, label="Fit")

for i in range(len(x)):
    if i==0:
        plt.plot([x[i], x[i]], [y[i], y_predicted[i]], color='tab:red', linestyle='-', label='Resiudals')

    plt.plot([x[i], x[i]], [y[i], y_predicted[i]], color='tab:red', linestyle='-')


plt.title("Single Variable Example", fontsize=13)
plt.xlabel(r"X: $RE(f_{0220})$ at BPM.11R1.B1", fontsize=13)
plt.ylabel(r"y: MQXA.1R1 $a_4$", fontsize=13)
plt.legend()



# %%
