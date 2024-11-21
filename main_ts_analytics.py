import sys 
import data_preprocessing as dp
import ts_analytics as ta

def main_ts_analytics():
    # define parameters 
    path = sys.argv[1]
    year = sys.argv[2] # name of the column year
    month = sys.argv[3] # name of the column month
    plot = sys.argv[4] # Anzahl or price (Volumen)
    m = int(sys.argv[5]) # season cycle
    
    # data processing
    ts = dp.upload_data_forecasting(path, plot, year, month)
    ts_train, ts_test = dp.prep_data_forecasting(ts, 0)
    
    # data analytics

    ta.plot_decomposition(ts_train, plot, m)



if __name__ == "__main__":
    main_ts_analytics()