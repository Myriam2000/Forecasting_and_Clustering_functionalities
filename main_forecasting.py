import sys 
import data_preprocessing as dp
import forecasting as f
import vizualisation as v

def main_forecasting():
    # define parameters 
    path = sys.argv[1]
    year = sys.argv[2] # name of the column year
    month = sys.argv[3] # name of the column month
    d = int(sys.argv[4]) # number of values to be predicted
    method = sys.argv[5].upper() # HW, ARMA, ARIMA, SARIMA
    plot = sys.argv[6] # Anzahl or price (Volumen)
    t = int(sys.argv[7]) # length of the test dataset
    m = int(sys.argv[8]) # season cycle
    
    #find input file name
    i = path.rfind('/')
    input_name = path[i+1:]
    
    # data processing
    ts = dp.upload_data_forecasting(path, plot, year, month)
    ts_train, ts_test = dp.prep_data_forecasting(ts, t)

    # forecasting
    forecast_output = f.use_forecasting_method(ts_train, ts, d, input_name, method, m)

    # vizualisation
    v.plot_forecast(forecast_output, ts_test, d, method)


if __name__ == "__main__":
    main_forecasting()