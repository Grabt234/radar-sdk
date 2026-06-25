import plotly.graph_objects as go
from IPython.display import display


class Generation:
    def __init__(self, title="Live Dynamic Plot", template="plotly_dark"):
        """Initializes a FigureWidget that leverages Plotly's native legend toggles."""
        self.fig = go.FigureWidget()

        # itemclick='toggle' means clicking a legend item shows/hides it.
        # itemdoubleclick='toggleothers' isolates a trace on double-click.
        self.fig.update_layout(
            title=title,
            template=template,
            showlegend=True,
            legend=dict(
                itemclick="toggle",  # Native checkbox-like behavior
                itemdoubleclick="toggleothers",  # Solo a trace on double-click
                orientation="h",  # Puts the legend horizontally
                yanchor="bottom",
                y=1.02,  # Places it neatly right above the plot
                xanchor="right",
                x=1,
            ),
        )

        # Internal storage
        self.streams = {}
        self.trace_indices = {}

    def _ensure_trace_exists(self, label):
        """Helper to create a new trace dynamically."""
        if label not in self.streams:
            self.streams[label] = {"x": [], "y": []}

            # Add a new scatter trace. It will automatically appear in the legend.
            self.fig.add_scatter(x=[], y=[], name=label, mode="lines+markers")
            self.trace_indices[label] = len(self.fig.data) - 1

    def append(self, label, x, y):
        """Appends a single data point to an existing stream and pushes to UI."""
        self._ensure_trace_exists(label)

        # 1. Update internal data cache
        self.streams[label]["x"].append(x)
        self.streams[label]["y"].append(y)

        # 2. Grab the specific trace index
        idx = self.trace_indices[label]

        # 3. Direct mutation of the FigureWidget trace updates the graph instantly
        self.fig.data[idx].x = self.streams[label]["x"]
        self.fig.data[idx].y = self.streams[label]["y"]

    def show(self):
        """Returns the figure widget to be rendered in Jupyter."""
        display(self.fig)
