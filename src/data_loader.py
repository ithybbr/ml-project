import pandas as pd

def load_data(path):
    df = pd.read_excel(path)
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    features = df.iloc[0]
    df.drop(index = 0, inplace=True)
    return df, features