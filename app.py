from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load your pretrained model
model = joblib.load('electricity_bill_predictor_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-bill', methods=['POST'])
def predict_bill():
    data = request.get_json()
    units = data.get('units')
    try:
        units = float(units)
        if units < 0:
            return jsonify({'error': 'Units must be a positive number'}), 400
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid units input'}), 400
    
    amount=model.predict([[units]])[0]
    # Tamil Nadu free units policy
    if units < 100:
        bill = 0.0
 
    else:
        bill = model.predict([[units]])[0] 

    return jsonify({'predicted_bill': round(bill, 2)})

if __name__ == '__main__':
    app.run(debug=True)
