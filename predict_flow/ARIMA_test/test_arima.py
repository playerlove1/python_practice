import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.metrics import mean_squared_error


#解析器  將'1-01'  對應為 1991 1月 以datetime的形式呈現
def parser(x):
	return pd.datetime.strptime('190'+str(x), '%Y-%m')

#找到最適合差分的階數 (只找前8階)  階數最小且p -value 小於0.05的
def best_diff(df, maxdiff = 8):
    p_set = []
    bestdiff = 0
    for i in range(0, maxdiff):
        temp=df.copy()
        if i>0:
            temp = temp.diff(i).dropna()
        p_set.append(ADF(temp.values)[1])
    i=0
    while i < len(p_set):
        if p_set[i] < 0.05:
            bestdiff = i
            break
        i += 1
    return bestdiff

def best_p_q(d, series):
    #經最佳差分後的資料
    d_series=series.diff(d).dropna()
    #一般而言階數不會超過 資料長度的 1/10
    pmax = int(len(d_series)/10)
    qmax = int(len(d_series)/10)
    #bic矩陣
    bic_matrix = []
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                tmp.append(ARIMA(series, order=(p,d,q)).fit(disp=0).bic)
            except:
                tmp.append(None)
            bic_matrix.append(tmp)
    
    #找出bic最小的 p,q參數
    bic_matrix = pd.DataFrame(bic_matrix) 
    p,q = bic_matrix.stack().idxmin() 
    return p,q
    
    
#arima 參數
p, d, q = 0,0,0 
    
#讀取時序資料 
series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#顯示前5筆資料
print(series.head())
#畫圖
series.plot()
#顯示圖
plt.show()

#檢驗該時間序列資料是否穩定  ADF檢定  (檢驗自迴歸模型是否存在 unit root 若有則代表為非平穩時間序列)
#H0:存在unit root H1:不存在unit root
# print('原始ADF:', ADF(series.values))
# print('ADF的p:', ADF(series.values)[1])
d = best_diff(series)
print("最佳差分階數:%d階" % d)

#檢驗時間序列是否滿足非隨機性  Ljung–Box 檢定  (若資料為不隨機分布則無法預測)
#H0:資料隨機分布  H1:資料不隨機分布
result=True if acorr_ljungbox(series.diff(d).dropna(), lags=1)[1] <0.05 else False
print("隨機性檢定結果(Ture 可分析, False 不可分析):",result)


#由於原時間序列是不定態的時間序列  為了使其穩定必須至少做一階差分
#因此利用自相關的圖來看大約需要幾個 lag  operator
#利用pandas 內建的自相關繪圖 畫出自相關圖
# autocorrelation_plot(series)
# plt.show()
#根據圖形判讀  約在前12 lag 為正相關
#前5個的 自相關係數皆大於0.5 因此可能是重要的
#而該AR模型  或許lag operatror =5是個不錯的起始點

#取得bic最佳的pq參數
p, q=best_p_q(d, series)

print("最佳參數組合ARIMA(%d,%d,%d)" %(p,d,q))


#使用最佳參數組合fit 
model = ARIMA(series, order=(p,d,q))
#由於在fit的過程中有很多線性迴歸的debug訊息出現 因此設定disp =0 關閉
model_fit = model.fit(disp=0)
#顯示ARIMA模型的摘要
print(model_fit.summary())

# plot residual errors (劃出殘差圖)
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
#殘差的密度分布 看似高斯分配 但中心不是0
residuals.plot(kind='kde')
plt.show()
#顯示詳細的殘差資訊
print(residuals.describe())

#取出時間序列的資料
X = series.values
#切割訓練集與測試集的資料大小
size = int(len(X) * 0.66)
#2/3的資料為訓練資料  1/3的資料為測試資料
train, test = X[0:size], X[size:len(X)]
#所有的歷史資料
history = [x for x in train]

#預測的list
predictions = list()

#針對所有的測試資料進行預測 ( one-step forecast)
for t in range(len(test)):
    model = ARIMA(history, order=(p,d,q))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    #持續加入新資料 以取得對下個時間段的預測
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

#計算MSE
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

#繪圖

#實際的資料
plt.plot(test)
#預測的資料
plt.plot(predictions, color='red')
plt.show()