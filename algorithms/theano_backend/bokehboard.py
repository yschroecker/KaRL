import bokeh
import bokeh.models
import bokeh.plotting
import bokeh.client
import bokeh.io
import bokeh.layouts
import time
import numpy as np
import collections


plot_width = 800
plot_height = 400


class _Bokehboard:
    class _SimplePlot:
        def __init__(self, title):
            self._data = bokeh.models.ColumnDataSource(data={'x': [], 'y': []})
            self.plot = bokeh.plotting.figure(title=title, plot_height=plot_height, plot_width=plot_width)
            self._line = self.plot.line(x=[], y=[])
            self._line.data_source = self._data

        def add_point(self, x, y):
            self._data.stream({'x': [x], 'y': [y]})

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

            self._quantile_data = {quantile: bokeh.models.ColumnDataSource(data={'xs': [], 'ys': []})
                                   for quantile in self._quantiles}
            self._patches = {quantile: self.plot.patches(xs=[], ys=[], line_alpha=0,
                                                         fill_alpha=0.4 - 0.2*(i/len(self._quantiles)))
                             for i, quantile in enumerate(self._quantiles)}
            self._last_quantile_y = {quantile: [0, 0] for quantile in self._quantiles}
            self._last_x = {quantile: 0 for quantile in self._quantiles}
            for quantile in self._quantiles:
                self._patches[quantile].data_source = self._quantile_data[quantile]

        def add_point(self, x, ys):
            self._mean_data.stream({'x': [x], 'y': [np.mean(ys)]})
            self._median_data.stream({'x': [x], 'y': [np.median(ys)]})
            for quantile in self._quantiles:
                last_y = self._last_quantile_y[quantile]
                y = [np.percentile(ys, 100-quantile), np.percentile(ys, quantile)]
                self._quantile_data[quantile].stream({'xs': [[self._last_x, x, x, self._last_x]],
                                                      'ys': [[last_y[0], y[0], y[1], last_y[1]]]})
                self._last_quantile_y[quantile] = y
            self._last_x = x

    LINE_PLOT = 0
    HISTOGRAM_PLOT = 1

    def __init__(self, session_name=None, plot_frequency=1):
        bokeh.io.curstate().autoadd = False
        self._watches = []
        self._python_vars = {}
        self._plot_frequency = plot_frequency
        self._update_counter = 0
        self._last_update = time.time()
        self._session_name = session_name

        self._session = None

    def _create_plot(self, description, plot_type):
        if plot_type == self.LINE_PLOT:
            plot = self._SimplePlot(description)
        elif plot_type == self.HISTOGRAM_PLOT:
            plot = self._HistogramPlot(description)
        else:
            assert False
        return plot

    def add_tensor_variable(self, description, var, plot_type):
        plot = self._create_plot(description, plot_type)
        self._watches.append((description, var, plot))

    def add_python_variable(self, description, plot_type, **kwargs):
        plot = self._create_plot(description, plot_type)
        for key, value in kwargs.items():
            self._python_vars[key] = [description, value, plot]

    def update_python_variable(self, **kwargs):
        for key, value in kwargs.items():
            self._python_vars[key][1] = value

    def show(self):
        doc = bokeh.plotting.curdoc()
        self._session = bokeh.client.push_session(doc, session_id=self._session_name)
        plots = [watch[2].plot for watch in self._watches]
        for key, (description, value, plot) in self._python_vars.items():
            plots.append(plot.plot)
        layout = bokeh.layouts.gridplot(plots, ncols=2)
        doc.add_root(layout)
        bokeh.client.show_session(self._session.id)

    def update(self):
        self._update_counter += 1
        current_time = time.time()
        if self._last_update + self._plot_frequency <= current_time:
            self._last_update = current_time
            self._update()

    def _update(self):
        for desc, var, plot in self._watches:
            value = var.get_value()
            plot.add_point(self._update_counter, value)
        for var, (desc, value, plot) in self._python_vars.items():
            plot.add_point(self._update_counter, value)

    def keep_alive(self):
        self._session.loop_until_closed()


class Bokehboard:
    instance = None

    def __init__(self):
        if not Bokehboard.instance:
            Bokehboard.instance = _Bokehboard()

    def __getattr__(self, name):
        return getattr(self.instance, name)


if __name__ == '__main__':
    import theano
    import time
    bb = Bokehboard()
    var = theano.shared([1, 1, 1], name="test")
    bb.add_tensor_variable("test", var, bb.HISTOGRAM_PLOT)
    bb.add_tensor_variable("test2", var, bb.HISTOGRAM_PLOT)
    bb.add_tensor_variable("test3", var, bb.HISTOGRAM_PLOT)
    bb.add_python_variable("test4", bb.LINE_PLOT, test4=3)
    bb.show()
    for i in range(1000):
        time.sleep(1)
        bb.update()
        var.set_value([i**k for k in np.arange(1, 3, 0.1)])
        bb.update_python_variable(test4=i)
    # bb._session.store_objects(bb._testplot._line.data_source)
    bb.keep_alive()

