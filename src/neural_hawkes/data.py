from dataclasses import dataclass
import numpy as np

@dataclass
class EventData:
    times: np.ndarray
    types: np.ndarray
    marks: np.ndarray

def load_csv(path: str) -> EventData:
    import pandas as pd
    df = pd.read_csv(path)
    return EventData(
        times=df["time"].to_numpy(),
        types=df["type"].to_numpy(),
        marks=df["mark"].to_numpy(),
    )