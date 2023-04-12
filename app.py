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

#to give some input
@app.route('/predict_api',methods =['POST'])
def predict_api():
    #converts the input data into json and store it on the data variable
    data = request.json['data']
    print(data)

    #getting only the value since it is json - key:value pair
    print(np.array(list(data.values())).reshape(1,-1)) 

    #stadardizing the giving input
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))

    #fitting the given data
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ =="__main__":
    app.run(debug = True)