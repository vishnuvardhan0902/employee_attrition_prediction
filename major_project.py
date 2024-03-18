from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib as jb
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def home():
    model = jb.load('attrition_model.joblib')
    df = pd.read_csv('finalDataset.csv')
    sc = StandardScaler()
    X = df.drop(['Attrition'], axis=1)
    y = df['Attrition']
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    score = model.score(X_test, y_test)
    df1 = df[['Age', 'DailyRate', 'DistanceFromHome', 'TotalWorkingYears', 'JobLevel', 'YearsSinceLastPromotion']]
    df_table = df1.head(10).to_html()
    return render_template('major_project.html', table=df_table, score=round(score, 4) * 100)

@app.route('/formdata', methods=['POST'])
def formdata():
    model = jb.load('attrition_model.joblib')
    df = pd.read_csv('finalDataset.csv')
    sc = StandardScaler()
    X = df.drop(['Attrition'], axis=1)
    y = df['Attrition']
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    data=request.form
    arr=[]
    for i in data.values():
        i=int(i)
        arr.append(i)
    age = arr[0]
    dist = arr[1]
    job_sat = arr[2]
    month_inc = arr[3]
    hike = arr[4]
    exp = arr[5]


    x_d=np.array([[age,802, dist,3, 1025, 6, 76, 3, 2, job_sat, 3, month_inc, 1,1,0,4,1,3, hike, 3, 3, 80, 0, exp, 3, 3, 7, 4, 0, 5, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
    x_d = sc.transform(x_d)
    
    y_pred = model.predict(x_d)
    if y_pred==1:
        y_p=  "Yes"
    else:
        y_p="No"
    return render_template('result.html', y_pred=y_p)

if __name__ == '__main__':
    app.run(debug=True)
