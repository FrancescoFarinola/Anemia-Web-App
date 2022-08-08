import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


scaler = StandardScaler()

conjreg = pd.read_csv("regression/smote_conj.csv")
Xc = conjreg.iloc[:, :-1].values
Yc = conjreg.iloc[:, -1].values
scaler.fit(Xc)
polpreg = pd.read_csv("regression/smote_polp.csv")
Xp = polpreg.iloc[:, :-1].values
Yp = polpreg.iloc[:, -1].values
scaler.fit(Xp)
lettoreg = pd.read_csv("regression/smote_letto.csv")
Xu = lettoreg.iloc[:, :-1].values
Yu = lettoreg.iloc[:, -1].values
scaler.fit(Xu)
testconj = pd.read_csv("regression/smote_conj_test.csv")
Xc_test = testconj.iloc[:, :-1].values
Yc_test = testconj.iloc[:, -1].values
scaler.fit(Xc_test)
testfinger = pd.read_csv("regression/smote_polp_test.csv")
Xp_test = testfinger.iloc[:, :-1].values
Yp_test = testfinger.iloc[:, -1].values
scaler.fit(Xp_test)
testnail = pd.read_csv("regression/smote_letto_test.csv")
Xu_test = testnail.iloc[:, :-1].values
Yu_test = testnail.iloc[:, -1].values
scaler.fit(Xu_test)

model1 = RandomForestRegressor(max_depth=7, n_estimators=50, n_jobs=1)
model1.fit(Xc,Yc)

model2 = RandomForestRegressor(max_depth=7, n_estimators=50, n_jobs=1)
model2.fit(Xp,Yp)

model3 = RandomForestRegressor(max_depth=7, n_estimators=50, n_jobs=1)
model3.fit(Xu, Yu)

scores1 = model1.predict(Xc_test)
scores2 = model2.predict(Xp_test)
scores3 = model3.predict(Xu_test)

print(scores1)
print(scores2)
print(scores3)

prediction = []
for i in range(len(scores1)):
    score = round((scores1[i]+scores2[i]+scores3[i])/3,2)
    prediction.append(score)
print(Yc_test)
print(prediction)
