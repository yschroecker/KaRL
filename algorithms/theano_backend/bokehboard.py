import bokeh
import bokeh.models
import bokeh.plotting
import bokeh.client


class Bokehboard:
    class _SimplePlot:
        def __init__(self, title):
            self._data = bokeh.models.ColumnDataSource(data={'x': [], 'y': []})
            self.plot = bokeh.plotting.figure(title=title, plot_height=300, plot_width=1000)
            self._line = self.plot.line(x=[], y=[])
            self._line.data_source = self._data

        def add_point(self, x, y):
            self._data.data = {'x': self._data.data['x'] + [x], 'y': self._data.data['y'] + [y]}

    LINE_PLOT = 0

    def __init__(self, session_name="default", plot_frequency=50):
        self._watches = []
        self._plot_frequency = plot_frequency
        self._update_counter = 0
        self._session_name = session_name

        self._doc = bokeh.plotting.curdoc()
        self._session = bokeh.client.push_session(self._doc, session_id=self._session_name)
        bokeh.client.show_session(self._session_name)

    def add_tensor_variable(self, description, var, plot_type):
        if plot_type == self.LINE_PLOT:
            plot = self._SimplePlot(description)
            self._doc.add_root(plot.plot)
        else:
            assert False
        self._watches.append((description, var, plot))

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
    var = theano.shared(1, name="test")
    bb.add_tensor_variable("test", var, Bokehboard.LINE_PLOT)
    for i in range(1000):
        time.sleep(1)
        bb.update()
        var.set_value(i**2)
    # bb._session.store_objects(bb._testplot._line.data_source)
    bb.keep_alive()

