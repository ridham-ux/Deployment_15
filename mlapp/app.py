from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

app = Flask(__name__)

# Load dataset for initial training
def load_data():
    data = pd.read_csv('housing.csv')
    return data

def train_model(data):
    # Prepare data for training
    X = data[['median_income', 'total_rooms', 'housing_median_age']]
    y = data['median_house_value']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        data = pd.read_csv(file)
        train_model(data)
        return render_template('result.html', result='Model trained successfully!')

@app.route('/manual', methods=['POST'])
def enter_data():
    median_income = float(request.form['median_income'])
    total_rooms = float(request.form['total_rooms'])
    housing_median_age = float(request.form['housing_median_age'])
    
    model = joblib.load('model.joblib')
    prediction = model.predict([[median_income, total_rooms, housing_median_age]])
    
    return render_template('result.html', result=f'Predicted median house value: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    # Load initial data and train model
    data = load_data()
    train_model(data)
    app.run(debug=True, host='0.0.0.0')
