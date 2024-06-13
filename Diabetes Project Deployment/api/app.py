import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

app.static_folder = 'static'

# Define the model class
class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.lin1 = nn.Linear(8, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.lin2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.lin3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.lin4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.lin1(x)))
        x = self.dropout2(self.relu2(self.lin2(x)))
        x = self.dropout3(self.relu3(self.lin3(x)))
        x = self.sigmoid(self.lin4(x))
        return x

# Instantiate the model
model = DiabetesModel()

# Load the trained model weights
model_path = os.path.join(os.path.dirname(__file__), 'diabeties_model_2.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from JSON
        data = request.get_json()
        input_data = [
            float(data['pregnancy']),
            float(data['glucose']),
            float(data['blood_pressure']),
            float(data['skin_thickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['pedigree_function']),
            float(data['age'])
        ]

        # Convert input data to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = (output >= 0.5).float().item()

        # Return prediction result
        result = "Positive for Diabetes. Check with a doctor!" if prediction == 1.0 else "Negative for Diabetes. May still want to consult a doctor."
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
