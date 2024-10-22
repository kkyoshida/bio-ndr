import figplot

figplot.plot_timeseries('ringtsne20-0')
figplot.plot_linsep_kl('ringtsne20-', data='ring')
figplot.plot_timeseries('ringtsne_simple20-0')
figplot.plot_linsep_kl('ringtsne_simple20-', data='ring')
figplot.plot_timeseries('mnist40-0')
figplot.plot_linsep_kl('mnist40-', data='MNIST')
figplot.plot_timeseries('mnist_simple40-0')
figplot.plot_linsep_kl('mnist_simple40-', data='MNIST')