import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('MacOSX')
import matplotlib.pyplot as plt

data = pd.read_csv('data/articles.csv')
plt.scatter(data['Article'], data['duration'])
plt.xlabel("Article")
plt.ylabel("Duration")
plt.show()

kmeans = KMeans(n_clusters=47, random_state=1).fit(data)
predicted_data=data.copy()
predicted_data['pred'] = kmeans.fit_predict(data)

plt.scatter(predicted_data["Article"], predicted_data['duration'], c=predicted_data['pred'], cmap='Blues')
plt.xlabel("Article")
plt.ylabel("Duration")
plt.show()