"""
Wiener-Hopf benchmark for Hawkes kernel estimation.

The estimated cross-covariance density G^{ij}(t) (t > 0) satisfies the
causal integral equation (Hawkes resolvent relation):

    G(t) = Phi(t) + integral_0^inf Phi(s) G(t-s) ds,   t > 0

Taking the one-sided Fourier transform (both Phi and G are causal):

    G_hat(w) = Phi_hat(w) (I + G_hat(w))
    =>
    Phi_hat(w) = G_hat(w) @ inv(I + G_hat(w))

Inverting back gives the estimated kernel matrix Phi(t) for t > 0.

Reference: Bacry, Mastromatteo & Muzy (2012),
           "Mean-Field Inference of Hawkes Processes".
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class WienerHopf:
    """
    Wiener-Hopf kernel estimator for multivariate Hawkes processes.

    Parameters
    ----------
    n_fft : int
        FFT size (should be a power of 2; larger => finer frequency resolution).
        Defaults to 65536.
    tikhonov : float
        Regularisation added to the diagonal of (I + G_hat) before inversion
        to improve numerical stability. Defaults to 1e-4.
    """

    def __init__(self, n_fft: int = 65536, tikhonov: float = 1e-4):
        self.n_fft = int(n_fft)
        self.tikhonov = float(tikhonov)
        self.phi_ = None        # estimated kernel (D, D, n_uniform)
        self.t_grid_ = None     # uniform time grid (n_uniform,)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        G: np.ndarray,
        t_grid: np.ndarray,
        t_max: float = None,
    ) -> "WienerHopf":
        """
        Estimate the kernel matrix from the empirical covariance density G.

        Parameters
        ----------
        G       : ndarray (D, D, n_t)
            Cross-covariance density estimated by StatisticEstimators.estimate_G,
            evaluated on t_grid (positive lags only).
        t_grid  : ndarray (n_t,)
            Non-uniform (hybrid lin/log) positive-lag grid matching G.
        t_max   : float, optional
            Truncation horizon for the output kernel.  Defaults to t_grid[-1].

        Returns
        -------
        self
        """
        G = np.asarray(G, dtype=float)
        t_grid = np.asarray(t_grid, dtype=float)
        D = G.shape[0]
        assert G.shape == (D, D, len(t_grid)), (
            f"G must have shape (D, D, n_t) = ({D}, {D}, {len(t_grid)})"
        )

        if t_max is None:
            t_max = float(t_grid[-1])

        n_uniform = self.n_fft // 2
        t_uniform = np.linspace(0.0, t_max, n_uniform + 1)[1:]  # avoid t=0
        dt = t_uniform[1] - t_uniform[0]

        # --- interpolate G onto a uniform grid --------------------------
        G_uniform = np.zeros((D, D, n_uniform), dtype=float)
        for i in range(D):
            for j in range(D):
                f = interp1d(
                    t_grid, G[i, j, :],
                    kind="linear",
                    bounds_error=False,
                    fill_value=(float(G[i, j, 0]), 0.0),
                )
                G_uniform[i, j, :] = f(t_uniform)

        # --- one-sided Fourier transform --------------------------------
        # G_hat(omega_k) ≈ dt * FFT(G_uniform)[k]
        G_fft_raw = np.fft.rfft(G_uniform, n=self.n_fft, axis=-1)  # (D, D, n_fft//2+1)
        G_hat = dt * G_fft_raw  # continuous FT approximation

        n_freq = G_hat.shape[-1]

        # --- Wiener-Hopf inversion: Phi_hat = G_hat @ inv(I + G_hat) ---
        Phi_hat = np.zeros_like(G_hat)
        I_reg = (1.0 + self.tikhonov) * np.eye(D, dtype=complex)

        for k in range(n_freq):
            Gk = G_hat[:, :, k]
            Phi_hat[:, :, k] = Gk @ np.linalg.inv(I_reg + Gk)

        # --- inverse Fourier transform ----------------------------------
        # phi(t_n) = IFFT(Phi_hat / dt)[n]
        phi_causal = np.fft.irfft(Phi_hat / dt, n=self.n_fft, axis=-1)

        # Keep only the positive-time part and clip negative values
        phi = phi_causal[:, :, :n_uniform]
        phi = np.maximum(phi, 0.0)

        self.phi_ = phi
        self.t_grid_ = t_uniform
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, t_query: np.ndarray) -> np.ndarray:
        """
        Evaluate the estimated kernel at arbitrary query times.

        Parameters
        ----------
        t_query : ndarray (n_q,)

        Returns
        -------
        phi_q : ndarray (D, D, n_q)
        """
        if self.phi_ is None:
            raise RuntimeError("Call fit() before predict().")
        t_query = np.asarray(t_query, dtype=float)
        D = self.phi_.shape[0]
        phi_q = np.zeros((D, D, len(t_query)), dtype=float)
        for i in range(D):
            for j in range(D):
                f = interp1d(
                    self.t_grid_, self.phi_[i, j, :],
                    kind="linear",
                    bounds_error=False,
                    fill_value=(float(self.phi_[i, j, 0]), 0.0),
                )
                phi_q[i, j, :] = np.maximum(f(t_query), 0.0)
        return phi_q

    def kernel_norms(self) -> np.ndarray:
        """
        Return the L1 norms ||phi^{ij}||_1 = integral phi^{ij}(t) dt,
        approximated by the trapezoidal rule on the uniform grid.

        Returns
        -------
        K : ndarray (D, D)
        """
        if self.phi_ is None:
            raise RuntimeError("Call fit() before kernel_norms().")
        dt = self.t_grid_[1] - self.t_grid_[0]
        return np.trapezoid(self.phi_, self.t_grid_, axis=-1)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        true_phi=None,
        t_max: float = None,
        n_points: int = 300,
    ):
        """
        Plot the estimated kernels, optionally overlaid with ground truth.

        Parameters
        ----------
        true_phi : callable or ndarray (D, D, n_t), optional
            Ground-truth kernel. If callable, called as true_phi(t) and must
            return a (D, D) or (D, D, n_t) array for a scalar or array t.
            If ndarray of shape (D, D, n_t), must be evaluated on t_grid_.
        t_max    : float, optional — truncation horizon for the plot.
        n_points : int — number of evaluation points.
        """
        if self.phi_ is None:
            raise RuntimeError("Call fit() before plot().")

        D = self.phi_.shape[0]
        t_max = t_max or float(self.t_grid_[-1])
        t_plot = np.linspace(self.t_grid_[0], t_max, n_points)
        phi_pred = self.predict(t_plot)  # (D, D, n_points)

        fig, axes = plt.subplots(
            D, D,
            figsize=(4.0 * D, 3.5 * D),
            sharex=True,
        )
        if D == 1:
            axes = np.array([[axes]])

        for i in range(D):
            for j in range(D):
                ax = axes[i, j]
                ax.plot(
                    t_plot, phi_pred[i, j, :],
                    color="#D95F02", lw=2.0,
                    label=rf"WH $\hat\phi^{{{i+1}{j+1}}}$",
                )

                if true_phi is not None:
                    if callable(true_phi):
                        phi_true_vals = np.array(
                            [true_phi(t)[i, j] for t in t_plot]
                        )
                    else:
                        phi_true_arr = np.asarray(true_phi, dtype=float)
                        f_true = interp1d(
                            self.t_grid_, phi_true_arr[i, j, :],
                            kind="linear",
                            bounds_error=False,
                            fill_value=0.0,
                        )
                        phi_true_vals = f_true(t_plot)
                    ax.plot(
                        t_plot, phi_true_vals,
                        color="#1C3F6E", lw=2.5, ls="--",
                        label=rf"True $\phi^{{{i+1}{j+1}}}$",
                    )

                ax.axhline(0, color="gray", lw=0.7, ls=":")
                ax.set_title(rf"$\phi^{{{i+1}{j+1}}}(t)$", fontsize=12)
                ax.set_xlim(0, t_max)
                ax.set_ylim(bottom=0)
                ax.legend(fontsize=9)
                ax.grid(ls="--", alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                if i == D - 1:
                    ax.set_xlabel("lag $t$")
                if j == 0:
                    ax.set_ylabel("kernel value")

        plt.suptitle(
            "Kernel matrix — Wiener-Hopf estimate",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
        return fig, axes


# ---------------------------------------------------------------------------
# Functional API (mirrors the usage style in the notebook)
# ---------------------------------------------------------------------------

def wiener_hopf_estimate(
    G: np.ndarray,
    t_grid: np.ndarray,
    t_max: float = None,
    n_fft: int = 65536,
    tikhonov: float = 1e-4,
) -> tuple:
    """
    One-shot Wiener-Hopf kernel estimation.

    Parameters
    ----------
    G        : ndarray (D, D, n_t) — empirical covariance density from estimate_G.
    t_grid   : ndarray (n_t,) — hybrid lin/log lag grid.
    t_max    : float, optional — output truncation horizon.
    n_fft    : int — FFT size.
    tikhonov : float — regularisation strength.

    Returns
    -------
    phi      : ndarray (D, D, n_uniform) — estimated kernel on a uniform grid.
    t_uniform: ndarray (n_uniform,) — the corresponding uniform time grid.
    model    : WienerHopf — fitted model (for predict / plot / kernel_norms).
    """
    model = WienerHopf(n_fft=n_fft, tikhonov=tikhonov)
    model.fit(G, t_grid, t_max=t_max)
    return model.phi_, model.t_grid_, model
