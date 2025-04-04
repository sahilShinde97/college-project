from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
GBModel = joblib.load('GBModelNormalData.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visual')
def visual():
    return render_template('visualisasi.html')

@app.route('/prediksi', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        input_data = request.form

        feature = [ 
            int(input_data['country']), 
            int(input_data['deposit_type']), 
            float(input_data['lead_time']), 
            int(input_data['total_of_special_requests']), 
            float(input_data['adr']), 
            int(input_data['market_segment']), 
            int(input_data['arrival_date_day_of_month']), 
            int(input_data['arrival_date_week_number']), 
            int(input_data['stays_in_week_nights']),
        ]

        # No scaling
        pred = GBModel.predict([feature])[0]
        pred_proba = GBModel.predict_proba([feature])
        prediction_result = f"{round(np.max(pred_proba) * 100, 2)}% {'(Cancel)' if pred == 1 else '(Not Cancel)'}"

        return render_template('prediksi.html', data=input_data, prediction=prediction_result,
            country=input_data['country'], deposit_type=input_data['deposit_type'], lead_time=input_data['lead_time'],
            total_of_special_requests=input_data['total_of_special_requests'], adr=input_data['adr'], 
            market_segment=input_data['market_segment'], arrival_date_day_of_month=input_data['arrival_date_day_of_month'], 
            arrival_date_week_number=input_data['arrival_date_week_number'], stays_in_week_nights=input_data['stays_in_week_nights']
        )

if __name__ == '__main__':
    app.run(debug=True)
