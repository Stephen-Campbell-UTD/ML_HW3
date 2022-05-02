# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
# %%

# (i)
df = pd.read_csv('./data/housing.csv').dropna()
y_all = df['median_house_value'].to_numpy(dtype=np.float32)
X_all = df.loc[:, df.columns !=
               'median_house_value'].to_numpy(dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=1)

# %%
# (ii)
min_samples = list(range(1, 26))
mse_array = []
mse_test_array = []
for i in min_samples:
    reg = tree.DecisionTreeRegressor(min_samples_leaf=i)
    reg = reg.fit(X_train, y_train)
    y_val_pred = reg.predict(X_val)
    y_test_pred = reg.predict(X_test)
    # squared -> False means RMSE
    mse_array.append(mean_squared_error(y_val, y_val_pred, squared=False))
    mse_test_array.append(mean_squared_error(
        y_test, y_test_pred, squared=False))

fig, ax = plt.subplots()
ax.plot(min_samples, mse_array)
ax.grid(True)
ax.set_ylabel("Mean Squared Error")
ax.set_xlabel("Min Samples per Leaf")
ax.set_title('Mean Squared Error vs. Min Samples per Leaf')

plt.savefig('./images/P1_ii_RMSEvsSamples')

# %%
# (iii)
print(min_samples[mse_array.index(min(mse_array))])
print(mse_test_array[mse_array.index(min(mse_array))])

# ~ 14 is the best miniumum number of leaf node observations

# %%

reg_to_cmp = [reg]
reg_to_cmp.append(linear_model.Ridge(alpha=0.5).fit(X_train, y_train))
reg_to_cmp.append(linear_model.ElasticNet().fit(X_train, y_train))
reg_to_cmp.append(
    linear_model.PassiveAggressiveRegressor().fit(X_train, y_train))
reg_to_cmp.append(svm.SVR().fit(X_train, y_train))


# for i in reg_to_cmp

indices = np.arange(len(reg_to_cmp))
mse_val = [mean_squared_error(y_val, reg.predict(
    X_val), squared=False) for reg in reg_to_cmp]
mse_test = [mean_squared_error(y_test, reg.predict(X_test), squared=False)
            for reg in reg_to_cmp]
# %%
barWidth = 0.35

fig, ax = plt.subplots(constrained_layout=True)
valRects = plt.bar(indices-barWidth/2, mse_val, barWidth,
                   color='b', label='Validation')
testRects = plt.bar(indices+barWidth/2, mse_test,
                    barWidth, color='g', label='Test')

ax.set_xlabel('Regressor')
ax.set_ylabel('Root Mean Squared Error')
ax.set_title("Root Mean Squared Error by Regressor")
ax.set_xticks(indices)
ax.set_xticklabels(
    ['Decision Tree', 'Ridge', 'ElasticNet', 'Passive\nAggressive', 'SVM']
)
ax.legend()
plt.savefig('./images/P1_iv_Comparison.png')
# fig.tight_layout()
# %%
