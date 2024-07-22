import pickle

from flask import Flask
from flask import request
from flask import jsonify


path = 'preprocess.bin'
loaded_model = 'model.pkl'

with open(path, 'rb') as f_in:
    dv = pickle.load(f_in)
    
    
with open(loaded_model, 'rb') as f_in:
    model = pickle.load(f_in)
    
app = Flask('Ride_duration_pred')

@app.route('/predict', methods = ['POST'])
def predict():
    client_pred = request.get_json()
    
    X = dv.transform([client_pred])

    ride_duration = model.predict(X)

    result = {
        'Predicted Ride Duration': float(ride_duration)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug = True, host = '127.0.0.1', port = 9090)