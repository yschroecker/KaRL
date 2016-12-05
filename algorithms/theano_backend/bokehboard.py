import bokeh
import bokeh.models
import bokeh.plotting
import bokeh.client
import bokeh.layouts
import numpy as np


plot_width = 800
plot_height = 400


class Bokehboard:
    class _SimplePlot:
        def __init__(self, title):
            self._data = bokeh.models.ColumnDataSource(data={'x': [], 'y': []})
            self.plot = bokeh.plotting.figure(title=title, plot_height=plot_height, plot_width=plot_width)
            self._line = self.plot.line(x=[], y=[])
            self._line.data_source = self._data

        def add_point(self, x, y):
            self._data.data = {'x': self._data.data['x'] + [x], 'y': self._data.data['y'] + [y]}

    class _HistogramPlot:
        _quantiles = [66, 80, 95]

        def __init__(self, title):
            self._mean_data = bokeh.models.ColumnDataSource(data={'x': [], 'y': []})
            self._median_data = bokeh.models.ColumnDataSource(data={'x': [], 'y': []})
            self.plot = bokeh.plotting.figure(title=title, plot_height=plot_height, plot_width=plot_width)
            self._mean_line = self.plot.line(x=[], y=[], line_width=1, line_dash=[3, 3])
            self._mean_line.data_source = self._mean_data
            self._median_line = self.plot.line(x=[], y=[], line_width=5)
            self._median_line.data_source = self._median_data

            self._quantile_data = {quantile: bokeh.models.ColumnDataSource(data={'x': [], 'y': []})
                                   for quantile in self._quantiles}
            self._patches = {quantile: self.plot.patch(x=[], y=[], fill_alpha=0.4 - 0.2*(i/len(self._quantiles)))
                             for i, quantile in enumerate(self._quantiles)}
            for quantile in self._quantiles:
                self._patches[quantile].data_source = self._quantile_data[quantile]

        def add_point(self, x, ys):
            self._mean_data.data.update(x=self._mean_data.data['x'] + [x],
                                        y=self._mean_data.data['y'] + [np.mean(ys)])
            self._median_data.data.update(x=self._median_data.data['x'] + [x],
                                          y=self._median_data.data['y'] + [np.median(ys)])
            for quantile in self._quantiles:
                self._quantile_data[quantile].data.update(x=[x] + self._quantile_data[quantile].data['x'] + [x],
                                                          y=[np.percentile(ys, 100-quantile)] +
                                                            self._quantile_data[quantile].data['y'] +
                                                            [np.percentile(ys, quantile)])

    LINE_PLOT = 0
    HISTOGRAM_PLOT = 1

    def __init__(self, session_name="default", plot_frequency=50):
        self._watches = []
        self._plot_frequency = plot_frequency
        self._update_counter = 0
        self._session_name = session_name

        self._session = None

    def add_tensor_variable(self, description, var, plot_type):
        if plot_type == self.LINE_PLOT:
            plot = self._SimplePlot(description)
        elif plot_type == self.HISTOGRAM_PLOT:
            plot = self._HistogramPlot(description)
        else:
            assert False
        self._watches.append((description, var, plot))

    def show(self):
        doc = bokeh.plotting.curdoc()
        self._session = bokeh.client.push_session(doc, session_id=self._session_name)
        layout = bokeh.layouts.gridplot([watch[2].plot for watch in self._watches], ncols=2)
        doc.add_root(layout)
        bokeh.client.show_session(self._session_name)

    def update(self):
        self._update_counter += 1

        if self._update_counter % self._plot_frequency == 0:
            self._update()

    def _update(self):
        for desc, var, plot in self._watches:
            value = var.get_value()
            plot.add_point(self._update_counter, value)

    def keep_alive(self):
        self._session.loop_until_closed()


if __name__ == '__main__':
    import theano
    import time
    bb = Bokehboard(plot_frequency=1, session_name="wha")
    var = theano.shared([1, 1, 1], name="test")
    bb.add_tensor_variable("test", var, Bokehboard.HISTOGRAM_PLOT)
    bb.add_tensor_variable("test2", var, Bokehboard.HISTOGRAM_PLOT)
    bb.add_tensor_variable("test3", var, Bokehboard.HISTOGRAM_PLOT)
    bb.show()
    for i in range(1000):
        time.sleep(1)
        bb.update()
        var.set_value([i**k for k in np.arange(1, 3, 0.1)])
    # bb._session.store_objects(bb._testplot._line.data_source)
    bb.keep_alive()

