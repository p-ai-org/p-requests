#!/usr/bin/env python3

from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from pathlib import Path
import os
import sys
import pandas as pd
import EDPipeline as edp

app = Flask(__name__)

#modelCD_path = os.path.join(Path(os.getcwd()).parents[1], 'p-requests','311_Requests_Model', 'modelCD.pkl')
#modelRT_path = os.path.join(Path(os.getcwd()).parents[1], 'p-requests','311_Requests_Model', 'modelRT.pkl')
#modelRF_path = os.path.join(Path(os.getcwd()).parents[1], 'p-requests','311_Requests_Model', 'randomForest.pkl')


modelCD = pickle.load(open('modelCD.pkl', 'rb'))
modelRT = pickle.load(open('modelRT.pkl', 'rb'))
modelRF = pickle.load(open('randomForest.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    features = [x for x in request.form.values()]
    features[-2] = float(features[-2])
    features[-1] = float(features[-1])
    print(features)
    column_names = ['AssignTo', 'RequestType', 'RequestSource', 'Month', 'Anonymous', 'CreatedByUserOrganization','Latitude','Longitude']
    dictionary = dict(zip(column_names,features))
    df_request = pd.DataFrame(columns= column_names)
    for key in dictionary: 
            df_request.at[0, key] = dictionary[key] 
    X, dfn = edp.preprocess_request(df_request)
    if(int(modelRF.predict(X)) == 0):
        return render_template('index.html', pred='More than 11 days')
    # averaged prediction --- pulled out from lacer script
    previous_fifty = np.load(open('previousfifty.npy','rb'))
    prediction = modelCD.predict(previous_fifty) + modelRT.predict(previous_fifty) / 2
    return render_template('index.html', pred=str(prediction[0]/24) + ' days')

if __name__ == '__main__':
    app.run(debug=True)
