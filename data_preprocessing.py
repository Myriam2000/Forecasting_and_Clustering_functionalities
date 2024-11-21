import pandas as pd


def remove_duplicate(dataframe):
    dataframe.drop_duplicates(keep = 'first', inplace=True)
    return dataframe

def optimize_types(X, df): # tranforms the df columns mentioned in X into numerical format 
    for num_col in X:
        df = df.replace({num_col: {',': '.'}}, regex=True)
        try :
            df[num_col]=pd.to_numeric(df[num_col], downcast="float")
        except:
            print('not numeric')
            quit()
    return df

def upload_data_clus(data_path, points, X):
    # importation des données
    columns = X.copy()
    columns.append(points)
    data = pd.read_csv(data_path, sep=';', encoding="latin_1", usecols = columns, index_col=points)
    # remove missing values
    data.dropna(inplace=True)
    # adapt the types
    data = optimize_types(X, data)
    return data

def prep_data_clus(data, points, X, func): 
    # creation of the dataset for clustering 
    if func=='mean':
        Cluster = data.groupby(points)[X].mean()
    if func=='sum':
        Cluster = data.groupby(points)[X].sum()
    if func=='count':
        Cluster = data.groupby(points)[X].count()
    return Cluster

def upload_data_forecasting(path, plot, year = 'Jahr', month='Monat'):
    columns=[year, month, plot]
    # extraction du dataset
    data = pd.read_csv(path, sep=';', encoding="latin_1", usecols=columns, parse_dates={'Date': [year, month]}, index_col='Date')
    # creation time series
    data = optimize_types([plot], data)
    ts = data[plot].groupby('Date').sum()
    return ts

def prep_data_forecasting(ts, t):
    if t != 0:
        ts_train = ts[:-t]
        ts_test = ts[-t:]
    else:
        ts_train = ts[-t:]
        ts_test = []
    return ts_train, ts_test