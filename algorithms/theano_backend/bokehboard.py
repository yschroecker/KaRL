import bokeh
import bokeh.models
import bokeh.models.widgets
import bokeh.plotting
import bokeh.client
import bokeh.io
import bokeh.layouts
import time
import numpy as np
import collections


plot_width = 800
plot_height = 400


class BoardBuilder:
    class _NetworkLayout:
        _Parameter = collections.namedtuple('_Parameter', ['parameter_name', 'parameter_plot', 'gradient_name',
                                                           'gradient_plot'])

        def __init__(self, board, name):
            self._board = board
            self.name = name
            self._parameters = {}

        def add_parameter(self, parameter_name, parameter_variable, gradient_name):
            parameter_plot = self._board.add_tensor_variable(parameter_name, parameter_variable,
                                                             self._board.HISTOGRAM_PLOT)
            gradient_plot = self._board.add_python_variable(gradient_name, self._board.HISTOGRAM_PLOT,
                                                            **{gradient_name:
                                                                   np.zeros_like(parameter_variable.get_value())})
            self._parameters[parameter_name] = \
                self._Parameter(parameter_name, parameter_plot, gradient_name, gradient_plot)
            self._board.add_tensor_variable(parameter_name, parameter_variable, self._board.HISTOGRAM_PLOT)

        def get_plots(self):
            return [pl for param in self._parameters.values() for pl in [param.parameter_plot, param.gradient_plot]]

        def get_parameter_names(self):
            return [param.parameter_name for param in self._parameters.values()]

    def __init__(self, board):
        self._networks = []
        self._board = board
        self._statistics = []

    def add_network(self, name):
        network_layout = self._NetworkLayout(self._board, name)
        self._networks.append(network_layout)
        return network_layout

    def add_statistic_pyvar(self, name, value):
        self._statistics.append(self._board.add_python_variable(name, self._board.LINE_PLOT, **{name: value}))

    def do_layout(self, session_name):
        doc = bokeh.plotting.curdoc()
        session = bokeh.client.push_session(doc, session_id=session_name)
        statistics_layout = bokeh.layouts.gridplot([plot.plot for plot in self._statistics], ncols=2)
        if len(self._statistics) == 0:
            tabs = []
        else:
            tabs = [bokeh.models.widgets.Panel(child=statistics_layout, title='Statistics')]
        for network in self._networks:
            layout = bokeh.layouts.gridplot([plot.plot for plot in network.get_plots()], ncols=2)
            tabs.append(bokeh.models.widgets.Panel(child=layout, title=network.name))
        doc.add_root(bokeh.models.widgets.Tabs(tabs=tabs))
        bokeh.client.show_session(session.id)


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
        _quantiles = [66, 80, 95, 100]

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
            self._last_x = 0
            for quantile in self._quantiles:
                self._patches[quantile].data_source = self._quantile_data[quantile]
            self._readjust_count = 0
            self._subsample = 1
            self._point_count = 0

        def add_point(self, x, ys):
            self._point_count += 1
            if self._point_count % self._subsample == 0:
                self._mean_data.stream({'x': [x], 'y': [np.mean(ys)]})
                self._median_data.stream({'x': [x], 'y': [np.median(ys)]})
                for quantile in self._quantiles:
                    last_y = self._last_quantile_y[quantile]
                    y = [np.percentile(ys, 100-quantile), np.percentile(ys, quantile)]
                    self._quantile_data[quantile].stream({'xs': [[self._last_x, x, x, self._last_x]],
                                                          'ys': [[last_y[0], y[0], y[1], last_y[1]]]})
                    self._last_quantile_y[quantile] = y
                self._last_x = x
                self._readjust_count += 1
                if self._readjust_count >= 100:
                    self._readjust()

        def _readjust(self):
            self._readjust_count = 10
            self._subsample *= 10
            # for quantile in self._quantiles:
            #     xs = []
            #     ys = []
            #     quantile_data = self._quantile_data[quantile].data
            #     for i in range(10, len(quantile_data['xs']), 10):
            #         xs.append([quantile_data['xs'][i - 10][0], quantile_data['xs'][i][0],
            #                    quantile_data['xs'][i][3], quantile_data['xs'][i - 10][3]])
            #         ys.append([quantile_data['ys'][i - 10][0], quantile_data['ys'][i][0],
            #                    quantile_data['ys'][i][3], quantile_data['ys'][i - 10][3]])
            #     xs.append([quantile_data['xs'][i][0], quantile_data['xs'][-1][1],
            #                quantile_data['xs'][-1][2], quantile_data['xs'][i][3]])
            #     ys.append([quantile_data['ys'][i][0], quantile_data['ys'][-1][1],
            #                quantile_data['ys'][-1][2], quantile_data['ys'][i][3]])
            #     self._quantile_data[quantile].data.update(xs=xs, ys=ys)


    LINE_PLOT = 0
    HISTOGRAM_PLOT = 1

    def __init__(self, session_name=None, plot_frequency=0.05):
        bokeh.io.curstate().autoadd = False
        self.board_builder = BoardBuilder(self)
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
        return plot

    def add_python_variable(self, description, plot_type, **kwargs):
        plot = self._create_plot(description, plot_type)
        for key, value in kwargs.items():
            self._python_vars[key] = [description, value, plot]
        return plot

    def update_python_variable(self, **kwargs):
        for key, value in kwargs.items():
            self._python_vars[key][1] = value

    def show(self):
        self.board_builder.do_layout(self._session_name)

    def update(self):
        self._update_counter += 1
        current_time = time.time()
        if self.ready_for_update():
            self._last_update = current_time
            self._update()

    def ready_for_update(self):
        return self._last_update + self._plot_frequency <= time.time()

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
    net1_var = theano.shared([1, 1, 1], name="net1.test")
    net2_var = theano.shared([1, 1, 1], name="net2.test")
    # bb.add_tensor_variable("test", var, bb.HISTOGRAM_PLOT)
    # bb.add_tensor_variable("test2", var, bb.HISTOGRAM_PLOT)
    # bb.add_tensor_variable("test3", var, bb.HISTOGRAM_PLOT)
    # bb.add_python_variable("test4", bb.LINE_PLOT, test4=3)
    bb.board_builder.add_statistic_pyvar("Q", 0)
    board_network = bb.board_builder.add_network("net1")
    board_network.add_parameter("net1.test", net1_var, "net1.test_gradient")
    board_network.add_parameter("net1.test2", net1_var, "net1.test2_gradient")
    board_network = bb.board_builder.add_network("net2")
    board_network.add_parameter("net2.test", net2_var, "net2.test_gradient")
    bb.show()
    for i in range(1000):
        time.sleep(1)
        bb.update()
        net1_var.set_value([i**k for k in np.arange(1, 3, 0.1)])
        net2_var.set_value([i*k for k in np.arange(1, 3, 0.1)])
        bb.update_python_variable(**{'net1.test_gradient': [i, 2*i, 3*i]})
        bb.update_python_variable(**{'net1.test2_gradient': [3*i, 2*i, 3*i]})
        bb.update_python_variable(**{'net2.test_gradient': [i, 2*i, 3*i]})
        bb.update_python_variable(Q=i)
        # if i == 3:
        #     value = [i**k for k in np.arange(1, 3, 0.1)]
        #     value[13] = np.float('inf')
        #     net1_var.set_value(value)
    # bb._session.store_objects(bb._testplot._line.data_source)
    bb.keep_alive()

