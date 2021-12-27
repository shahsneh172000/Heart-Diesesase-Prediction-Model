from flask import Flask , request , render_template
import ml_model

app = Flask(__name__)

@app.route("/")
def home():
    var = ml_model.l
    return render_template('heart_index.html')

@app.route("/heart_pred",methods=['POST'])
def heart_pred():
    in_data = [x for x in request.form.values()]

    age = in_data[0]
    sex = in_data[1]
    bp = in_data[2]
    chol = in_data[3]
    h_rate = in_data[4]
    fbs = in_data[5]
    oldpeak = in_data[6]
    pain = in_data[7]
    ecg = in_data[8]
    angina = in_data[9]
    slope = in_data[10]
    
    ans = ml_model.pred(age,bp,chol,fbs,h_rate,oldpeak,sex,pain,ecg,angina,slope)
    return render_template('pred.html',pred_text = "Chances of Heart Disease = "+str(ans) + " % ",data = in_data)

app.run(debug=True)