#lets import the necessary libraries
import numpy as np
from flask import Flask, render_template, request
import pickle

#load the model file
model = pickle.load(open('model.pkl','rb'))

#lets initialize the flask app
app = Flask(__name__)

#lets define our default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#to use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    #for rendering the results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))

#lets start flask server
if __name__ == '__main__':
    app.run(debug=True)

