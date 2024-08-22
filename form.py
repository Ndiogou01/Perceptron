from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
perceptron = joblib.load('perceptron.pkl')

@app.route('/')
def home():
    return render_template('Prediction.html')
@app.route('/prediction', methods=['POST'])
def prediction():
    sepal_length = float(request.form['sepalLength'])
    petal_length = float(request.form['petalLength'])
    predict = perceptron[0] + sepal_length * perceptron[1] + petal_length * perceptron[2]
    if predict > 0 :
        return render_template('Iris_Versicolor.html')
    else:
        return render_template('Iris_Setosa.html')
if __name__ == '__main__':
    app.run(port=5001)