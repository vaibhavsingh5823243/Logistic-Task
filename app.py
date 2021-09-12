from flask import Flask,request,render_template,jsonify
from main import HandlingMultipleFiles,DataPreprocessing,Models
import pickle
import time
file_obj=HandlingMultipleFiles(dirname='AReM')
data=file_obj.mergefile()
data_preprocessing=DataPreprocessing(data=data)
model_obj=Models()
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict/',methods=['POST','GET'])
def form():
    if request.method=='POST':
        avr_rss12=float(request.form['avgrss12'])
        var_rss12=float(request.form['varrss12'])
        avr_rss13 = float(request.form['avgrss13'])
        var_rss13 = float(request.form['varrss13'])
        avr_rss23 = float(request.form['avgrss23'])
        var_rss23 = float(request.form['varrss23'])
        feature=[[avr_rss12,var_rss12,avr_rss13,var_rss13,avr_rss23,var_rss23]]
        with open('scaler.pickle','rb') as f:
            scaler=pickle.load(f)
            feature=scaler.transform(feature)
        result=model_obj.predict(feature)
        #result=data_preprocessing.change_target(int(result[0]),num_to_cat=True)
        feature=data['Lable'].unique()
        return f"{feature[result]}" #f"<p>{result}</p>"
    return render_template('form.html')
@app.route('/visualization/')
def visualize():
    try:
        return render_template('profile_report.html')
    except Exception as e:
        return e


if __name__=='__main__':
    app.run()
