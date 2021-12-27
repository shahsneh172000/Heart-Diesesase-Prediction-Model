import numpy as np
import pandas as pd
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data//clean_heart_data.csv",)

x = df.drop('HeartDisease',axis=1)
y = df['HeartDisease']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3)

sc = StandardScaler()
q = sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

# lgr = LogisticRegression()
# lgr.fit(x_train,y_train)

l = list(x.columns)
l = l[1:]
ml_model = joblib.load(r"heart_ml_model.pkl")
# print(l[:])

def pred(age,bp,cholestroel,fasting_bs,max_hr,oldpeak,sex,chest_pain,resting_ecg,angina,st_slope):
    
    arr = np.zeros(len(x.columns))
    
    arr[0] = age
    arr[1] = bp
    arr[2] = cholestroel
    arr[3] = fasting_bs
    arr[4] = max_hr
    arr[5] = oldpeak
    
    if "Sex_" + sex in x.columns:
        index = np.where(x.columns=="Sex_"+sex)
        arr[index]=1
        
    if "ChestPainType_" + chest_pain in x.columns:
        index = np.where(x.columns=="ChestPainType_"+chest_pain)
        arr[index]=1
        
    if "RestingECG_" + resting_ecg in x.columns:
        index = np.where(x.columns=="RestingECG_"+resting_ecg)
        arr[index]=1
    
    if "ExerciseAngina_" + angina:
        index = np.where(x.columns=="ExerciseAngina_"+angina)
        arr[index]=1
        
    if "ST_Slope_" + st_slope in x.columns:
        index = np.where(x.columns=="ST_Slope_"+st_slope)
        arr[index]=1
  
    arr = sc.transform([arr])
    coef = ml_model.coef_
    coef = coef.reshape(20)
    s = sum(coef*arr[0])+ ml_model.intercept_[0]
    ans = 1/(1+2.718**(-1*s))
    #print("Prob=",1/(1+2.718**(-1*s)))
    return int(ans*100)

print(pred(48,127,329,0,120,1,'M','TA','ST','N','Flat'))