import plotly.graph_objects as go
from IPython.display import display

class Generation:
    def __init__(self, title="Live Dynamic Plot", template="plotly_dark"):
        """Initializes an empty FigureWidget to handle multiple live data streams."""
        self.fig = go.FigureWidget()
        self.fig.update_layout(title=title, template=template)
        
        # Internal storage to keep track of data points per stream label
        # Structure: { "stream_name": {"x": [], "y": []} }
        self.streams = {}
        
        # Maps a stream name to its index in the figure's data tuple
        self.trace_indices = {}
        
    def _ensure_trace_exists(self, label):
        """Helper to create a new trace dynamically if the label hasn't been used yet."""
        if label not in self.streams:
            self.streams[label] = {"x": [], "y": []}
            
            # Add a new scatter trace to the Plotly figure
            self.fig.add_scatter(x=[], y=[], name=label, mode='lines+markers')
            
            # Record the index of this new trace
            self.trace_indices[label] = len(self.fig.data) - 1

    def append(self, label, x, y):
        """Appends a single data point (or small lists) to an existing stream."""
        self._ensure_trace_exists(label)
        
        # Append to our local memory bank
        self.streams[label]["x"].append(x)
        self.streams[label]["y"].append(y)
        
        # Instantly push the updated lists to the specific Plotly trace
        idx = self.trace_indices[label]
        self.fig.data[idx].x = self.streams[label]["x"]
        self.fig.data[idx].y = self.streams[label]["y"]

    def set_data(self, label, x_array, y_array):
        """Overwrites or bulk-loads an entire array of data for a stream."""
        self._ensure_trace_exists(label)
        
        self.streams[label]["x"] = list(x_array)
        self.streams[label]["y"] = list(y_array)
        
        idx = self.trace_indices[label]
        self.fig.data[idx].x = self.streams[label]["x"]
        self.fig.data[idx].y = self.streams[label]["y"]

    def clear(self):
        """Resets all internal data buffers and clears the visual chart traces."""
        self.streams.clear()
        self.trace_indices.clear()
        self.fig.data = []  # Completely wipes all traces out of the figure layout

    def show(self):
        """Displays the embedded live widget in the notebook cell."""
        display(self.fig)