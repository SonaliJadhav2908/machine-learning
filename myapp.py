import numpy as np
from flask import Flask,render_template,request
import pickle
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

@app.route('/')
def hello_world():
   return render_template('form.html')

@app.route('/predict',methods = ['POST'])

def get_result():
  poly = pickle.load(open('Poly.pkl','rb'))
  model = pickle.load(open('model.pkl','rb' ))
  query = [[float(request.form['text2'])]] 
  X_query=poly.transform(query)
  sal = model.predict(X_query)

  return 'Dear '+request.form["text1"] + 'Your predicted salary is '+request.form ["text2"]+'Experience is :'+str(sal)

if __name__=='__main__':
    app.run(debug=True)