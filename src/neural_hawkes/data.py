from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class EventData:
    times: np.ndarray
    types: np.ndarray
    marks: np.ndarray
    marks_binned: Optional[np.ndarray] = None
    horizon: Optional[float] = None

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.types = np.asarray(self.types, dtype=int)
        self.marks = np.asarray(self.marks, dtype=float)

        if self.horizon is None:
            self.horizon = float(self.times.max()) if len(self.times) > 0 else 0.0

    @property
    def n_events(self) -> int:
        return len(self.times)

    def sort_by_time(self):
        idx = np.argsort(self.times)
        self.times = self.times[idx]
        self.types = self.types[idx]
        self.marks = self.marks[idx]
        if self.marks_binned is not None:
            self.marks_binned = self.marks_binned[idx]

    def attach_binned_marks(self,marks_binned:np.ndarray):
        if len(marks_binned) != self.n_events:
            raise ValueError("Length of marks_binned is different from the number of events.")
        self.marks_binned = np.asarray(marks_binned, dtype=int)

    def validate(self, D: int):
        if not (len(self.times) == len(self.types) == len(self.marks)):
            raise ValueError("times = types = marks Needed ! ")
        if np.any(self.times < 0):
            raise ValueError("Event times should be >0 !")
        if np.any(self.types < 0) or np.any(self.types >= D):
            raise ValueError(f"Event types must be in {{0, ..., {D-1}}}")
        if len(self.times) > 1 and np.any(np.diff(self.times) < 0):
            raise ValueError("Event times must be sorted")

def load_csv(path: str) -> EventData:

    df = pd.read_csv(path)

    events = EventData(
        times=df["time"].to_numpy(),
        types=df["type"].to_numpy(),
        marks=df["mark"].to_numpy(),
    )
    events.sort_by_time()
    return events
