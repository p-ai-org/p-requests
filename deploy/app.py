#!/usr/bin/env python3

from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from pathlib import Path
import os

app = Flask(__name__)

modelCD_path = os.path.join(Path(os.getcwd()).parents[1], '311_Requests_Model', 'modelCD.pkl')
modelRT_path = os.path.join(Path(os.getcwd()).parents[1], '311_Requests_Model', 'modelRT.pkl')

modelCD = pickle.load(open(modelCD_path, 'rb'))
modelRT = pickle.load(open(modelRT_path, 'rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    # averaged prediction --- pulled out from lacer script
    prediction = modelCD.predict(final) + modelRT.predict(final) / 2
    return render_template('index.html', pred=str(prediction))

if __name__ == '__main__':
    app.run(debug=True)
