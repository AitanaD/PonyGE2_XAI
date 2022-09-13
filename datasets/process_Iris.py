import numpy as np
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

path = "Iris/Iris.csv"
df=pd.read_csv(path)
print(df.head)

train, test = train_test_split(df, test_size=0.2)

#filename = "Banknote.csv"

#urllib.request.urlretrieve(url, filename)

#data = np.genfromtxt(filename, delimiter=",")

np.random.seed(0)
#np.random.shuffle(data)

# whenever the last column is < 0.5 (ie 0), change it to 0
#data[data[:,-1] < 0.5,-1] = -1

#train = data[:1000]
#test = data[1000:]
np.savetxt('Iris/Train.csv', train, delimiter=" ", header="x0 x1 x2 x3 y", fmt='%s')
np.savetxt('Iris/Test.csv', test, delimiter=" ", header="x0 x1 x2 x3 y", fmt='%s')
