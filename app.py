# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 14:24:59 2021

@author: Yashasvee Shukla
"""

from flask import Flask, render_template, request
import pickle
import numpy as np



app = Flask(__name__)
model = pickle.load(open('iris.pk1', 'rb'))



@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)