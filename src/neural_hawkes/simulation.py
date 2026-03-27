from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd
from .data import EventData


ArrayLike = np.ndarray


def exponential_mark_sampler(rng: np.random.Generator, scale: float = 1.0) -> float:
    return float(rng.exponential(scale=scale))


def identity_mark_impact(mark: float) -> float:
    return float(mark)


def clipped_exp_mark_impact(mark: float, gamma: float = 0.3, clip: float = 10.0) -> float:
    """
    phi(mark) = exp(gamma * mark), clipped for stability.
    """
    return float(np.clip(np.exp(gamma * mark), 0.0, clip))


@dataclass
class MarkedExpHawkesParams:
    """
    Intensity:
        lambda_i(t) = mu_i + sum_j sum_{T_n^j < t} alpha_{ij} * phi(M_n) * exp(-beta_{ij} (t-T_n))
    """
    mu: ArrayLike
    alpha: ArrayLike
    beta: ArrayLike

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float)
        self.alpha = np.asarray(self.alpha, dtype=float)
        self.beta = np.asarray(self.beta, dtype=float)

        if self.mu.ndim != 1:
            raise ValueError("mu must have shape [D].")
        D = self.mu.shape[0]

        if self.alpha.shape != (D, D):
            raise ValueError("alpha must have shape [D, D].")
        if self.beta.shape != (D, D):
            raise ValueError("beta must have shape [D, D].")
        if np.any(self.mu < 0):
            raise ValueError("mu must be nonnegative.")
        if np.any(self.alpha < 0):
            raise ValueError("alpha must be nonnegative for this simulator.")
        if np.any(self.beta <= 0):
            raise ValueError("beta must be strictly positive.")

    @property
    def D(self) -> int:
        return self.mu.shape[0]


class MarkedExpHawkesSimulator:
    def __init__(
        self,
        params: MarkedExpHawkesParams,
        mark_sampler: Optional[Callable[[np.random.Generator], float]] = None,
        mark_impact: Optional[Callable[[float], float]] = None,
        seed: Optional[int] = None):
        self.params = params
        self.mark_sampler = mark_sampler or exponential_mark_sampler
        self.mark_impact = mark_impact or identity_mark_impact
        self.rng = np.random.default_rng(seed)

    def _decay_excitation(self, excitation: np.ndarray, dt: float) -> np.ndarray:
        """
        Exponential decay of each excitation component over dt.
        """
        return excitation * np.exp(-self.params.beta * dt)

    def _current_intensity(self, excitation: np.ndarray) -> np.ndarray:
        """
        lambda_i(t) = mu_i + sum_j excitation[i, j]
        """
        return self.params.mu + excitation.sum(axis=1)

    def simulate(self,horizon: float,max_events: int = 100_000) -> EventData:
        """
        Simulate events up to time `horizon`.

        Returns
        -------
        EventData
            times, types, marks arrays
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        D = self.params.D
        t = 0.0

        # excitation[i, j] = current contribution to intensity i from source type j
        excitation = np.zeros((D, D), dtype=float)

        times: list[float] = []
        types: list[int] = []
        marks: list[float] = []

        n_events = 0

        while t < horizon and n_events < max_events:
            lambda_now = self._current_intensity(excitation)
            lambda_bar = float(lambda_now.sum())

            if lambda_bar <= 0:
                break

            dt = self.rng.exponential(scale=1.0 / lambda_bar)
            t_candidate = t + dt

            if t_candidate > horizon:
                break

            excitation_candidate = self._decay_excitation(excitation, dt)
            lambda_candidate_vec = self._current_intensity(excitation_candidate)
            lambda_candidate_sum = float(lambda_candidate_vec.sum())

            accept_prob = lambda_candidate_sum / lambda_bar
            if self.rng.uniform() <= accept_prob:
   
                probs = lambda_candidate_vec / lambda_candidate_sum
                event_type = int(self.rng.choice(D, p=probs))


                mark = float(self.mark_sampler(self.rng))
                jump_scale = float(self.mark_impact(mark))

                # After accepting the event, add alpha[:, event_type] * phi(mark)
                excitation = excitation_candidate.copy()
                excitation[:, event_type] += self.params.alpha[:, event_type] * jump_scale

                t = t_candidate
                times.append(t)
                types.append(event_type)
                marks.append(mark)
                n_events += 1
            else:
                # Reject; move time forward with decayed excitation
                t = t_candidate
                excitation = excitation_candidate

        return EventData(
            times=np.asarray(times, dtype=float),
            types=np.asarray(types, dtype=int),
            marks=np.asarray(marks, dtype=float),
            horizon=float(horizon),
        )

    def simulate_many(self,n_paths: int,horizon: float,max_events: int = 100_000) -> list[EventData]:
        return [self.simulate(horizon=horizon, max_events=max_events) for _ in range(n_paths)]


def save_events_to_csv(events: EventData, path: str) -> None:
    df = pd.DataFrame(
        {
            "time": events.times,
            "type": events.types,
            "mark": events.marks,
        }
    )
    df.to_csv(path, index=False)


def default_toy_simulator(seed: int = 42) -> MarkedExpHawkesSimulator:
    params = MarkedExpHawkesParams(
        mu=np.array([0.25, 0.20]),
        alpha=np.array([
            [0.18, 0.08],
            [0.06, 0.16],
        ]),
        beta=np.array([
            [1.4, 1.0],
            [1.1, 1.6],
        ]),
    )

    return MarkedExpHawkesSimulator(params=params,mark_sampler=lambda rng: exponential_mark_sampler(rng, scale=1.0),
        mark_impact=lambda m: clipped_exp_mark_impact(m, gamma=0.15, clip=4.0),seed=seed)


if __name__ == "__main__":
    sim = default_toy_simulator(seed=42)
    events = sim.simulate(horizon=30.0, max_events=50_000)
    save_events_to_csv(events, "data/events.csv")
    print(f"Saved {events.n_events} events to data/events.csv")