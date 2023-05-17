import pandas as pd

from sklearn.preprocessing import normalize

datasets = ["splice.csv", "balance-scale.csv", "cmc.csv", "vehicle.csv"]

for d in datasets:
    print("Dataset", d)

    df = pd.read_csv(d)
    y = df[df.columns[-1]]

    # Transform categorical values
    dummy_df = pd.get_dummies(df.drop(df.columns[-1], axis=1))

    #Save header values
    features = list(dummy_df)
    print(features)

    #Normalize data by columns
    new_df = normalize(dummy_df)
    new_df = pd.DataFrame(new_df, columns=features)
    new_df[df.columns[-1]] = y



    print(df.head())
    print(new_df.head())

    #Save new datafrmae
    new_df.to_csv("new_" + d)
