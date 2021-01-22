import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv('heart.csv')

DataToScale = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
Xprocessed = StandardScaler().fit_transform(DataToScale)
kme = KMeans(n_clusters=2, max_iter=300)
kme.fit(Xprocessed)
Yval = kme.predict(Xprocessed)

#print(kme.labels_)
#55 incorrect out of 303
#81.8% accuracy

DataToScale['cluster'] = Yval
DataToScale.cluster.unique()

p1 = DataToScale[DataToScale.cluster == 0]
p2 = DataToScale[DataToScale.cluster == 1]


#In these scatter plots, Cluster 0 represents no disease, and cluster 1 represents disease.
#I have commented out these scatter plots, as they were saved as images to be used in the project.
#The code has been kept for review and for creating new scatter plots if desired

#plt.scatter(p1['age'], p1['chol'], color = 'red', label = 'No disease')
#plt.scatter(p2['age'], p2['chol'], color = 'blue', label = 'Disease')
#plt.xlabel('Age')
#plt.ylabel('Cholesterol level - mg/dL')
#plt.legend()
#plt.show()

#plt.scatter(p1['age'], p1['thalach'], color = 'red', label = 'No disease')
#plt.scatter(p2['age'], p2['thalach'], color = 'blue', label = 'Disease')
#plt.xlabel('Age')
#plt.ylabel('Maximum Heart Rate')
#plt.legend()
#plt.show()

#plt.scatter(p1['trestbps'], p1['thalach'], color = 'red', label = 'No disease')
#plt.scatter(p2['trestbps'], p2['thalach'], color = 'blue', label = 'Disease')
#plt.xlabel('Resting Blood Pressure')
#plt.ylabel('Maximum Heart Rate')
#plt.legend()
#plt.show()