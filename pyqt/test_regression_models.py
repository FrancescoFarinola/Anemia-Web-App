import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

scaler= StandardScaler()
perf = pd.DataFrame(columns=['Tissue','Algorithm','MAE','MSE','R2'])

conjreg = pd.read_csv("regression train/int_conj.csv")
Xc = conjreg.iloc[:, :-1].values
Yc = conjreg.iloc[:, -1].values
polpreg = pd.read_csv("regression train/smote_polp.csv")
Xp = polpreg.iloc[:, :-1].values
Yp = polpreg.iloc[:, -1].values
lettoreg = pd.read_csv("regression train/smote_letto.csv")
Xu = lettoreg.iloc[:, :-1].values
Yu = lettoreg.iloc[:, -1].values
unicoreg = pd.read_csv("regression train/manual_unico.csv")
X = unicoreg.iloc[:, :-1].values
Y = unicoreg.iloc[:, -1].values
scaler.fit(Xc)
scaler.fit(Xp)
scaler.fit(Xu)
scaler.fit(X)
kf = KFold(n_splits=10, shuffle=True)

#TEST CONGIUNTIVA
mae, mse, r2 = [], [], []
for i in range(50):
    model1 = RandomForestRegressor(max_depth=7, n_estimators=50, n_jobs=1)
    results = cross_validate(model1, Xc, Yc, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Conjunctiva', 'Algorithm': 'Random Forest', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model2 =Ridge(alpha=0.1)
    results = cross_validate(model2, Xc, Yc, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Conjunctiva', 'Algorithm': 'Ridge Linear', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model3 = SVR(kernel='rbf', gamma='auto')
    results = cross_validate(model3, Xc, Yc, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Conjunctiva', 'Algorithm': 'SVR rbf', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model4 = SVR(kernel='linear')
    results = cross_validate(model4, Xc, Yc, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Conjunctiva', 'Algorithm': 'SVR Linear', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

predicted = cross_val_predict(model1, Xc, Yc, cv=10)
fig, ax = plt.subplots()
ax.scatter(Yc, predicted, edgecolors=(0, 0, 0))
ax.plot([Yc.min(), Yc.max()], [Yc.min(), Yc.max()], 'k--', lw=4)
ax.set_xlabel('Reale')
ax.set_ylabel('Predetto')
ax.set_title('Congiuntiva - Integrale - SVR linear')
plt.savefig('Conjintegrale.png')
plt.show()

#TEST POLPASTRELLO
mae, mse, r2 = [], [], []
for i in range(10):
    model1 = RandomForestRegressor(max_depth=7, n_estimators=50, n_jobs=1)
    results = cross_validate(model1, Xp, Yp, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Fingertip', 'Algorithm': 'Random Forest', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model2 =Ridge(alpha=0.1)
    results = cross_validate(model2, Xp, Yp, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Fingertip', 'Algorithm': 'Ridge Linear', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model3 = SVR(kernel='rbf', gamma='auto')
    results = cross_validate(model3, Xp, Yp, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Fingertip', 'Algorithm': 'SVR rbf', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model4 = SVR(kernel='linear')
    results = cross_validate(model4, Xp, Yp, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Fingertip', 'Algorithm': 'SVR Linear', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

#TEST LETTO UNGUEALE
mae, mse, r2 = [], [], []
for i in range(10):
    model1 = RandomForestRegressor(max_depth=7, n_estimators=50, n_jobs=1)
    results = cross_validate(model1, Xu, Yu, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Nailbed', 'Algorithm': 'Random Forest', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model2 =Ridge(alpha=0.1)
    results = cross_validate(model2, Xu, Yu, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Nailbed', 'Algorithm': 'Ridge Linear', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model3 = SVR(kernel='rbf', gamma='auto')
    results = cross_validate(model3, Xu, Yu, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Nailbed', 'Algorithm': 'SVR rbf', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model4 = SVR(kernel='linear')
    results = cross_validate(model4, Xu, Yu, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Nailbed', 'Algorithm': 'SVR Linear', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

#TEST UNICO
mae, mse, r2 = [], [], []
for i in range(50):
    model1 = RandomForestRegressor(max_depth=7, n_estimators=50, n_jobs=1)
    results = cross_validate(model1, X, Y, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Unico', 'Algorithm': 'Random Forest', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model2 =Ridge(alpha=0.1)
    results = cross_validate(model2, X, Y, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Unico', 'Algorithm': 'Ridge Linear', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

predicted = cross_val_predict(model1, X, Y, cv=10)
fig, ax = plt.subplots()
ax.scatter(Y, predicted, edgecolors=(0, 0, 0))
ax.plot([Yc.min(), Yc.max()], [Yc.min(), Yc.max()], 'k--', lw=4)
ax.set_xlabel('Reale')
ax.set_ylabel('Predetto')
ax.set_title('Classificatore unico bilanciamento manuale - Ridge Linear')
plt.savefig('Unicomanuale.png')
plt.show()

mae, mse, r2 = [], [], []
for i in range(10):
    model3 = SVR(kernel='rbf', gamma='auto')
    results = cross_validate(model3, X, Y, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Unico', 'Algorithm': 'SVR rbf', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

mae, mse, r2 = [], [], []
for i in range(10):
    model4 = SVR(kernel='linear')
    results = cross_validate(model4, X, Y, cv=kf,
                             scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    mae.append(results['test_neg_mean_absolute_error'].mean())
    mse.append(results['test_neg_mean_squared_error'].mean())
    r2.append(results['test_r2'].mean())
row = {'Tissue': 'Unico', 'Algorithm': 'SVR Linear', 'MAE': abs(np.array(mae).mean()), 'MSE': abs(np.array(mse).mean()), 'R2': np.array(r2).mean()}
perf = perf.append(row, ignore_index=True)
del mae, mse, r2

print(perf)
export_excel = perf.to_excel(r'C:\Users\farin\Desktop\export_dataframe1.xlsx', index = None, header=True)