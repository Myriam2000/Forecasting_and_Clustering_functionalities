from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def plot_decomposition(ts_train, plot, period=12):
    result=seasonal_decompose(ts_train, model='multiplicable', period=period)
    trend_estimate = result.trend
    seasonal_estimate = result.seasonal
    residual_estimate = result.resid
    # Plotting the time series and it's components together
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(6)
    fig.set_figwidth(10)
    # First plot to the Original time series
    axes[0].plot(ts_train, label='Original') 
    axes[0].legend(loc='upper left');
    # second plot to be for trend
    axes[1].plot(trend_estimate, label='Trend')
    axes[1].legend(loc='upper left');
    # third plot to be Seasonality component
    axes[2].plot(seasonal_estimate, label='Seasonality')
    axes[2].legend(loc='upper left');
    # last last plot to be Residual component
    axes[3].plot(residual_estimate, label='Residuals')
    axes[3].legend(loc='upper left');
    plt.savefig('../images/decomposition_%s.png'%plot)