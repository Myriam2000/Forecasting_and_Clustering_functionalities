import matplotlib.pyplot as plt

def plot_forecast(forecast_output, ts_test, d, method):
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_output['data'][:-d], label = 'training data')
    plt.plot(forecast_output['fitting'][1:-d], label = 'fitting')
    if len(ts_test) != 0:
        plt.plot(ts_test, label = 'testing data')
    plt.plot(forecast_output['data'][-d:], label = 'predictions')
    plt.fill_between(forecast_output.index[-d:], forecast_output['lower'][-d:], forecast_output['upper'][-d:], color='k', alpha=.2)
    plt.title('%s forecasting'%method)
    plt.legend(loc ="upper left")
    plt.savefig('../images/f_plot_%s.png'%method) 


def plot_clus_2D(X, points, output, method):# plot if 2D
    plt.figure(figsize=(6,4))
    plt.scatter(output.values[:,0], output.values[:, 1], s=10, c=output.values[:,2])
    plt.xlabel(X[0])
    plt.ylabel(X[1])
    plt.title("{} : {}".format(method, points))
    plt.savefig('../images/clustering_plot_{}_{}.png'.format(method, points))