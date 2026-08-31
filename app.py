from flask import Flask, render_template, request, redirect, url_for, session, send_file
from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

app.secret_key = "warehouse_secret"

# MONGODB

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["warehouse_ai_db"]

prediction_collection = db["predictions"]

db = client["warehouse_ai_db"]

prediction_collection = db["predictions"]

# LOAD MODEL

model = joblib.load("warehouse_model.pkl")

encoder = joblib.load("label_encoder.pkl")

# USERS
users = {

    "admin": "admin123",

    "manager": "manager123",

    "staff": "staff123"
}

# HOME
@app.route('/')
def home():

    return redirect(url_for('login'))

# LOGIN
@app.route('/login', methods=['GET', 'POST'])
def login():

    error = None

    if request.method == 'POST':

        username = request.form['username']

        password = request.form['password']

        if username in users:

            if users[username] == password:

                session['username'] = username

                return redirect(url_for('predict'))

            else:
                error = "Wrong Password"

        else:
            error = "Invalid Username"

    return render_template(
        'login.html',
        error=error
    )

# PREDICT PAGE
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    prediction = None

    reason = None

    if request.method == 'POST':

        category = request.form['category']

        zone = request.form['zone']

        stock_level = float(request.form['stock_level'])

        reorder_point = float(request.form['reorder_point'])

        turnover_ratio = float(request.form['turnover_ratio'])

        kpi_score = float(request.form['kpi_score'])

        stockout = float(request.form['stockout'])

        input_data = np.array([[
            stock_level,
            reorder_point,
            turnover_ratio,
            kpi_score,
            stockout
        ]])

        result = model.predict(input_data)

        prediction = encoder.inverse_transform(result)[0]

        # REASON TEXT
        if prediction == "High Risk":

            reason = (
                "Stock level is lower than reorder point "
                "and inventory stockout is high. "
                "Immediate restocking recommended."
            )

        elif prediction == "Medium Risk":

            reason = (
                "Inventory is moderately stable "
                "but stockout count requires monitoring."
            )

        else:

            reason = (
                "Inventory levels are stable "
                "and KPI score is healthy."
            )

        # STORE IN MONGODB
        prediction_collection.insert_one({

            "username": session['username'],

            "category": category,

            "zone": zone,

            "stock_level": stock_level,

            "reorder_point": reorder_point,

            "turnover_ratio": turnover_ratio,

            "kpi_score": kpi_score,

            "stockout": stockout,

            "prediction": prediction,

            "reason": reason
        })

    return render_template(
        'predict.html',
        prediction=prediction,
        reason=reason
    )

# ANALYTICS
@app.route('/analytics')
def analytics():

    data = list(prediction_collection.find())

    df = pd.DataFrame(data)

    category_chart = {}

    zone_chart = {}

    if not df.empty:

        category_chart = (
            df['category']
            .value_counts()
            .to_dict()
        )

        zone_chart = (
            df['zone']
            .value_counts()
            .to_dict()
        )

    return render_template(
        'analytics.html',
        category_chart=category_chart,
        zone_chart=zone_chart
    )

# =========================
# HISTORY
# =========================

@app.route('/history')
def history():

    data = list(prediction_collection.find())

    return render_template(
        'history.html',
        data=data
    )

# =========================
# DOWNLOAD REPORT
# =========================

@app.route('/download/<filetype>')
def download(filetype):

    data = list(prediction_collection.find())

    df = pd.DataFrame(data)

    if '_id' in df.columns:

        df.drop('_id', axis=1, inplace=True)

    if not os.path.exists("reports"):

        os.makedirs("reports")

    # CSV
    if filetype == "csv":

        path = "reports/report.csv"

        df.to_csv(path, index=False)

        return send_file(
            path,
            as_attachment=True
        )

    # EXCEL
    elif filetype == "excel":

        path = "reports/report.xlsx"

        df.to_excel(path, index=False)

        return send_file(
            path,
            as_attachment=True
        )

    # JSON
    elif filetype == "json":

        path = "reports/report.json"

        df.to_json(
            path,
            orient='records'
        )

        return send_file(
            path,
            as_attachment=True
        )

# LOGOUT
@app.route('/logout')
def logout():

    session.clear()

    return redirect(url_for('login'))

# RUN
if __name__ == '__main__':

    app.run(debug=True)