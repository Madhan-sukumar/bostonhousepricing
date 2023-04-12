import pickle
import json
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#load the pickled model
regmodel = pickle.load(open('regmodel.pkl','rb'))
#loading the scaling for standardisation
scalar=pickle.load(open('scaling.pkl','rb'))

#go to the home page
@app.route('/')
def home():
    return render_template('home.html')

#to give some input and test the api on POSTMAN
@app.route('/predict_api',methods =['POST'])
def predict_api():
    #converts the input data into json and store it on the data variable using request object
    data = request.json['data']
    print(data)

    #getting only the value since it is json - key:value pair
    print(np.array(list(data.values())).reshape(1,-1)) 

    #stadardizing the giving input
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))

    #fitting the given data
    output = regmodel.predict(new_data)

    #since it return 2d array and need 1st value
    print(output[0])
    return jsonify(output[0])

#to capture and transform all values in front end application
@app.route('/predict',methods=['POST'])
def predict():

    #to capture all the values using form in html and request object. convert into as list using list comprehension
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    
    #since it is a 2d array and need 1st value
    output = regmodel.predict(final_input)[0]

    #using render template, replace the predicted value in the place holder "prediction_text" on home.html  
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))


if __name__ =="__main__":
    app.run(debug = True)