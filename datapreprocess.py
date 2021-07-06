import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

label_map={"0":"ANGRY","1":"HAPPY","2":"SAD","3":"SURPRISE","4":"NEUTRAL"}

df = pd.read_csv("./ferdata.csv")

df.head()

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.1,random_state=42,stratify =dataY)