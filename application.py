
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('Cleaned_car_data.csv')
app = Flask(__name__)
@app.route("/")
def show():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())
    return render_template('web_index.html', companies = companies, car_models= car_models, years = years, fuel_types = fuel_types)


@app.route('/predict',methods=['POST'])

def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kilo_driven'))

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))

    print(prediction)

    return str(np.round(prediction[0],2))





if __name__ == "__main__":
    app.run(debug=True)