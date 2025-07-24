from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model_performance.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # simple form page

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    study_hours = float(request.form['study_hours'])
    attendance = int(request.form['attendance_percentage'])
    prev_grade = int(request.form['previous_grade_numeric'])  # e.g., A=0, B=1...

    # Prepare input as a list of list for prediction
    input_features = [[study_hours, attendance, prev_grade]]

    # Make prediction
    prediction = model.predict(input_features)

    # Optional: Convert encoded prediction back to label if needed
    label_map = {0: 'Average', 1: 'Poor', 2: 'Good'}
    predicted_label = label_map.get(prediction[0], "Unknown")

    return render_template('result.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
