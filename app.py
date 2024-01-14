from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('Dragon.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = []

        for feature_name in [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]:
            features.append(float(request.form[feature_name]))

        features = np.array([features])
        prediction = model.predict(features)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
