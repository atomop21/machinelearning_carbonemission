#import the required libraries
from pyexpat import model
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("FuelConsumption.csv")

#lets extract the features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#lets prepare the data for training
x = cdf.iloc[: , 0:3]
y = cdf.iloc[:, 3]
print("x=",x)
print("y=",y)

regressor = LinearRegression()

#lets train the model using the algorithm
regressor.fit(x,y)

#saving the model
#Pickle serializes objects so they can be 
#saved to file, and loaded in a program later on.
pickle.dump(regressor, open('model.pkl','wb'))

#lets predict the output
#load the model file
'''model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))'''