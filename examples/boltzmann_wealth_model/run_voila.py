# %matplotlib widget
import threading
import time

import ipywidgets as widgets
import matplotlib.pyplot as plt
from boltzmann_wealth_model.model import BoltzmannWealthModel
from IPython.display import display
from matplotlib.ticker import MaxNLocator


def agent_portrayal(content):
    # assert len(content) <= 1
    if len(content) == 0:
        return 0
    agent = content[0]
    if agent.wealth > 0:
        return 2
    return 1


def make_button(name):
    return widgets.Button(
        description=name,
        disabled=False,
        button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
        # tooltip=name,
        # icon="check",  # (FontAwesome names without the `fa-` prefix)
    )


class JupyterVisualization:
    def __init__(self, model_class, width_height, measures=None, name="Mesa Model"):
        self.model_class = model_class
        self.width, self.height = width_height
        self.measures = measures
        self.log_output = widgets.Output()
        self.thread = None
        self.thread_flag = threading.Event()
        button_step = make_button("Step")
        button_start = make_button("Start")
        button_stop = make_button("Stop")
        button_reset = make_button("Reset")
        button_step.on_click(self.do_step)
        button_reset.on_click(self.reset_plots)
        button_start.on_click(self.threaded_do_start)
        button_stop.on_click(self.do_stop)

        self.slider_n_agents = widgets.IntSlider(
            value=10,
            min=1,
            max=100,
            step=1,
            description="Number of agents:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )

        label_name = widgets.Label(value=name)

        gs_buttons = widgets.GridspecLayout(n_rows=1, n_columns=5)
        gs_buttons[0, 0] = label_name
        gs_buttons[0, 1] = button_step
        gs_buttons[0, 2] = button_start
        gs_buttons[0, 3] = button_stop
        gs_buttons[0, 4] = button_reset

        gs_ctrl = widgets.GridspecLayout(n_rows=1, n_columns=1)
        gs_ctrl[0, 0] = self.slider_n_agents

        self.gs_plots = widgets.GridspecLayout(n_rows=len(measures) + 1, n_columns=1)

        al = widgets.AppLayout(
            header=gs_buttons,
            left_sidebar=gs_ctrl,
            center=self.gs_plots,
            right_sidebar=self.log_output,
            footer=None,
            pane_heights=["35px", 10, 0],
        )

        display(al)

        # Grid
        with plt.ioff():
            self.space_fig, self.space_ax = plt.subplots(1, 1)
        # Remove title
        self.space_fig.canvas.header_visible = False
        self.space_fig.canvas.toolbar_visible = False
        self.gs_plots[0, 0] = self.space_fig.canvas

        self.initialize_plots()

    @staticmethod
    def portray(g):
        return [
            [agent_portrayal(subcontent) for subcontent in content] for content in g
        ]

    def initialize_space(self):
        self.imshow_object = self.space_ax.imshow(self.portray(self.model.grid._grid))

    def draw_space(self):
        self.imshow_object.set_data(self.portray(self.model.grid._grid))
        self.space_fig.canvas.clear_output(wait=True)
        self.space_fig.canvas.draw()
        self.space_fig.canvas.flush_events()

    def initialize_plots(self):
        self.plots = []
        for i, measure in enumerate(self.measures):
            with plt.ioff():
                fig, ax = plt.subplots(1, 1)
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            (line,) = plt.plot([], [])
            ax.set_ylabel(measure)
            # Set integer x axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            self.gs_plots[i + 1, 0] = fig.canvas
            self.plots.append([fig, ax, line])
        self.reset_plots(at_initialization=True)

    def draw_plot(self, plot_obj, df, name):
        fig, ax, line = plot_obj
        xs = list(range(len(df)))
        ys = df.loc[:, name]
        line.set_data(xs, ys)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def reset_plots(self, event=None, at_initialization=False):
        self.model = self.model_class(
            self.slider_n_agents.value, self.width, self.height
        )
        self.initialize_space()
        if at_initialization:
            return
        for obj in self.plots:
            self.draw_plot(obj, {}, None)
        with self.log_output:
            print("Reset!")

    def do_step(self, event):
        self.model.step()
        df = self.model.datacollector.get_model_vars_dataframe()
        tic = time.time()
        for i, measure in enumerate(self.measures):
            self.draw_plot(self.plots[i], df, measure)
        self.draw_space()
        self.log_output.clear_output(wait=True)
        with self.log_output:
            print("Elapsed", round(time.time() - tic, 3))

    def do_start(self):
        self.model.running = True
        #self.thread_flag.clear()
        #while not self.thread_flag.is_set() and self.model.running:
        while self.model.running:
            self.do_step(None)

    def threaded_do_start(self, event):
        self.thread = threading.Thread(target=self.do_start)
        self.thread.start()

    def do_stop(self, event):
        #self.thread_flag.set()
        # self.thread.join()
        self.model.running = False
        self.thread.join()


viz = JupyterVisualization(
    BoltzmannWealthModel, (10, 10), measures=["Gini"], name="Money Model"
)
