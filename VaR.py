
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[258]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[525]:


from scipy import stats
import statsmodels.api as sm


# ### Описание задачи
#    * Оценить VaR на горизонте 10 дней от вложений в зерно на основе моделирования поведения факторов.
# 

# ### Импорт данных

# Импортируем данные за 2 года и посмотрим на них 

# In[468]:


brent = pd.read_csv("Brent_Oil_Futures_Historical_Data.csv")
dj = pd.read_csv("Dow_Jones_Industrial_Average_Historical_Data.csv")
tapr = pd.read_csv("TAPR_Historical_Data.csv")
wheat = pd.read_csv("Thomson_Reuters_CoreCommodity_CRB_Wheat_Historical_Data.csv")


# In[469]:


print("Brent:")
print(brent.describe())
print(brent.columns)
print("Dow Jones:")
print(dj.describe())
print(dj.columns)
print("TAPR:")
print(tapr.describe())
print(tapr.columns)
print("Wheat:")
print(wheat.describe())
print(wheat.columns)


# Для анализа будем использовать столбцы Date и Price.
# 
# Приводим числовые данные - цену Dow Jones - к нужному виду

# In[470]:


dj.Price = dj.Price.apply(lambda x: x.replace(',','')).astype(float)
dj.describe()


# #### Объединение данных в одну таблицу по дате

# In[228]:


required_columns = ["Date", "Price"]
data = dj[required_columns]
for df,suffix in zip([brent, tapr, wheat], ["_brent", "_tapr", "_wheat"]):
    data = pd.merge(data, df[required_columns], on="Date", how="outer", suffixes=("", suffix))
    
data.rename({"Price": "Price_dj"}, axis='columns', inplace=True)


# In[241]:


data.head()


# Приведём дату к нужному типу данных и отсортируем по дате.

# In[266]:


data.Date = data.Date.apply(lambda date: pd.datetime.strptime(date, "%b %d, %Y"))
data.sort_values(by='Date', inplace=True)
data.Date.head()


# Посмотрим на изменение цен с течением времени

# In[481]:


plt.figure(figsize=(25,30))
plt.subplot(4,1,1)
plt.fill_between(data.Date.values, data.Price_dj, alpha=0.3, color='C0')
plt.plot(data.Date.values, data.Price_dj, color='C0')
plt.grid()
plt.title("Dow Jones")
plt.subplot(4,1,2)
plt.fill_between(data.Date.values, data.Price_brent, alpha=0.3, color='C1')
plt.plot(data.Date.values, data.Price_brent, color='C1')
plt.grid()
plt.title("Brent")
plt.subplot(4,1,3)
plt.fill_between(data.Date.values, data.Price_tapr, alpha=0.3, color='C2')
plt.plot(data.Date.values, data.Price_tapr, color='C2')
plt.grid()
plt.title("TAPR")
plt.subplot(4,1,4)
plt.fill_between(data.Date.values, data.Price_wheat, alpha=0.3, color='C3')
plt.plot(data.Date.values, data.Price_wheat, color='C3')
plt.grid()
plt.title("Wheat")
plt.show()


# ### Заполнение пропущенных значений

# Посмотрим на строки с пропущенными значениями

# In[245]:


data[data.isna().apply(any, axis=1)]


# Пропущенные значения - в дни национальных праздников и некоторые другие дни - заполним последним непропущенным значением для фактора, т. е. действительной ценой к тому дню, в который у нас нет значения

# In[269]:


data.fillna(method="ffill", inplace=True)


# In[275]:


data.apply(lambda x: all(x.isna()))


# Теперь, когда нет пропущенных значений, преобразуем данные - посчитаем 10-дневную доходность для каждого фактора (в долях)

# In[290]:


def count_profit(column, period=10):
    return (column[period:].values - column[:-period].values)/column[:-period].values
 
profit = pd.DataFrame()
profit["brent"] = count_profit(data.Price_brent)
profit["dj"] = count_profit(data.Price_dj)
profit["tapr"] = count_profit(data.Price_tapr)
profit["wheat"] = count_profit(data.Price_wheat)


# In[435]:


profit.head()


# На этих данных будет обучаться модель

# ### Линейная модель зависимости инструмента от факторов

# Распределения и взаимосвязи между переменными в выборке

# In[444]:


pd.plotting.scatter_matrix(profit, figsize=(15,15))


# Разобьём данные на обучающую и тестовую выборку

# In[301]:


predictors = ["dj", "brent", "tapr"]
X_train, X_test, y_train, y_test = train_test_split(profit[predictors], profit["wheat"], test_size=0.25, shuffle=False)


# Построим модель - линейную регрессию

# In[302]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[447]:


print("Ошибка на обучающей выборке: {:.6}".format(mean_squared_error(model.predict(X_train), y_train)))
print("Ошибка на тестовой выборке: {:.6}".format(mean_squared_error(model.predict(X_test), y_test)))


# In[450]:


print(model.coef_)
print(model.intercept_)


# #### Анализ модели

# In[457]:


X = sm.add_constant(X_train) 
statmodel = sm.OLS(y_train, X)
results = statmodel.fit()


# In[458]:


print(results.summary())


# Выводы: 
#  * маленький коэффициент детерминации говорит о том, что построенная модель не улавливает изменения целевой переменной. Для более корректного анализа необходимо использовать другую модель.
#  * коэффициенты при dj и brent значимы на уровне alpha = 0.05

#   ### Оценка параметров распределений факторов и генерация значений 

# Оценим необходимые параметры распределений факторов по имеющейся выборке: выборочное среднее, дисперсию, корреляции (ковариации).

# In[354]:


means = profit[predictors].apply(stats.tmean)
stds = profit[predictors].apply(stats.tstd)
covs = profit[predictors].cov()


# Создадим распределение и сгенерируем N = 500 новый значений

# In[459]:


factors_distr = stats.multivariate_normal(means, covs)


# In[482]:


N = 500
np.random.seed(42)
random_factors = factors_distr.rvs(size=N)


# Вычислим значение wheat с помощью построенной линейной модели

# In[483]:


prediction = model.predict(random_factors)


# ### Оценка значений VaR и CVaR

# Оценим значения VaR и CVar по получившемуся эмпирическому распределению

# In[484]:


def get_VaR(factor):
    return np.percentile(prediction,1)
VaR = get_VaR(prediction)
print("VaR: {:.6}".format(VaR))


# In[485]:


def get_CVaR(factor, VaR=None):
    if not(VaR):
       VaR = get_VaR(factor) 
    below_var = prediction[prediction <= VaR]
    return np.mean(below_var)
CVaR = get_CVaR(prediction, VaR)
print("CVaR: {:.6}".format(CVaR))


# Изобразим получившиеся значения на графике

# In[524]:


below_var = prediction[prediction <= VaR]
plt.figure(figsize=(15,12))
ax = plt.subplot(2,1,1)
plt.hist(prediction,bins=15)
plt.hist(below_var, bins=2)
plt.grid()
plt.title("Generated wheat distribution")
ticks = ax.get_xticks()
tickls = ax.get_xticklabels()
ax.set_xticks(np.append(ticks,[VaR, CVaR]))
ax.set_xticklabels(np.append(np.round(ticks,2),["VaR", "CVaR"]))
plt.subplot(2,1,2)
plt.bar(x=range(len(prediction)),height=np.sort(prediction))
plt.bar(x=range(len(below_var)),height=np.sort(below_var))
plt.grid()
plt.title("Wheat ordered sample")
plt.show()


# ### Оценка вероятностного интервала значений VaR и CVaR

# Так как мы не предполагаем ничего о распределении инструмента wheat, сгенерируем много выборок размера N, на каждой выборке получим ответ модели и значения VaR и CVaR. Затем построим эмпирические плотности распределений для этих параметров и по ним рассчитаем доверительный интервал. 

# In[418]:


M = 500 # количество симуляций
np.random.seed(0)
VaRs = np.zeros(M)
CVaRs = np.zeros(M)
for i in range(M):
    factors = factors_distr.rvs(N)
    prediction = model.predict(factors)
    VaR = get_VaR(prediction)
    CVaR = get_CVaR(prediction, VaR)
    VaRs[i] = VaR
    CVaRs[i] = CVaR
    


# In[462]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.hist(VaRs)
plt.title("VaR distribution")
plt.subplot(1,2,2)
plt.hist(CVaRs)
plt.title("CVaR distribution")
plt.show()


# In[430]:


alpha = 0.01
print("Вероятностный {:d}% интервал для VaR: [{:.6}, {:.6}]".format(
    round((1-alpha)*100), np.percentile(VaRs, alpha/2), np.percentile(VaRs, 1-alpha/2)))
print("Вероятностный {:d}% интервал для CVaR: [{:.6}, {:.6}]".format(
    round((1-alpha)*100), np.percentile(CVaRs, alpha/2), np.percentile(CVaRs, 1-alpha/2)))

