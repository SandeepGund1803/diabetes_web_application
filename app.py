from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('diabetes.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    Glucose = request.form['Glucose']
    BloodPressure = request.form['BloodPressure']
    SkinThickness = request.form['SkinThickness']
    Insulin = request.form['Insulin']
    BMI = request.form['BMI']
    DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    Age = request.form['Age']

    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    arr = np.array([[Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    df1 = pd.DataFrame(arr)

    std_scaler1 = StandardScaler()
    std_array1 = std_scaler1.fit_transform(df1)
    


    pred = model.predict(std_array1)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)   

# @app.route('/predict1')
# def predict1():
#     a = int(request.args.get('a'))
#     b = int(request.args.get('b'))
#     c = int(request.args.get('c'))
#     d = int(request.args.get('d'))

#     arr = np.array([[a, b, c, d]])
#     pred = model.predict(arr)
#     return render_template('after.html', data=pred)