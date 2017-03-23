import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Imputer


train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);
           
plt.show()