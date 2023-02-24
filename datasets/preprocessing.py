import pandas as pd

from sklearn.preprocessing import normalize

datasets = ["blood-transfusion.csv", "climate-model.csv", "ilpd.csv", "ozone-level-8hr.csv"]

for d in datasets:
    print("Dataset", d)

    df = pd.read_csv(d)
    y = df[df.columns[-1]]

    #Save header values
    features = list(df.drop(df.columns[-1], axis=1))
    print(features)

    #Normalize data by columns
    new_df = normalize(df.drop(df.columns[-1], axis=1))
    new_df = pd.DataFrame(new_df, columns=features)
    new_df[df.columns[-1]] = y

    print(df.head())
    print(new_df.head())

    #Save new datafrmae
    new_df.to_csv("new_" + d)
