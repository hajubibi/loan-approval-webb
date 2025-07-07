from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ''
    if request.method == 'POST':
        Gender = 1 if request.form['Gender'] == 'Male' else 0
        Married = 1 if request.form['Married'] == 'Yes' else 0
        Dependents = int(request.form['Dependents'])
        Education = 1 if request.form['Education'] == 'Graduate' else 0
        Self_Employed = 1 if request.form['Self_Employed'] == 'Yes' else 0
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = float(request.form['Credit_History'])
        Property_Area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[request.form['Property_Area']]

        input_data = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                                ApplicantIncome, CoapplicantIncome, LoanAmount,
                                Loan_Amount_Term, Credit_History, Property_Area]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)