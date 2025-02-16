from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the trained model
with open("model/student_performance_model.pkl", "rb") as f:
    scaler, label_encoder, model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        features = [float(request.form[key]) for key in ["math_score", "reading_score", "writing_score"]]
        features = np.array(features).reshape(1, -1)
        
        # Scale input data
        features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features)
        race_ethnicity = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'prediction': race_ethnicity})

    except ValueError:
        return jsonify({'error': 'Invalid input! Enter numeric values only.'})

if __name__ == "__main__":
    app.run(debug=True)
