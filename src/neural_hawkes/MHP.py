import numpy as np
import matplotlib.pyplot as plt


class MHP:
    """
    D-dimensional linear Hawkes process
    The conditional intensity of component i is:
        \lambda^i_t = \mu^i  +  \sum_j  \sum_{t_k^j < t}  \phi^{ij}(t − t_k^j)
    where  \Phi = (\phi^{ij})_{1≤i,j≤D}  is the kernel matrix.

    Parametric mode (default)
    Pass Phi as a (D×D) array of floats \alpha^{ij}
    Kernels are exponential:  \phi^{ij}(t) = \alpha^{ij} · \omega · exp(−\omega·t)

    Custom kernel mode
    Pass Phi as a (D×D) array of callables \phi^{ij} : R_+ → R_+

    Parameters
    ----------
    mu    : (D,) array of baseline intensities \mu^i
    omega : float — exponential decay rate \omega (parametric)
    T_max : float — integration horizon for stability check in custom mode
    """

    def __init__(self, Phi=[[0.5]], mu=[0.1], omega=1.0, T_max=50.0):
        self.mu = np.asarray(mu, dtype=float)
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D array.")
        if np.any(self.mu < 0):
            raise ValueError("mu must be nonnegative.")

        self.D = self.mu.shape[0]
        self.omega = float(omega)
        self.T_max = float(T_max)

        Phi_arr = np.array(Phi, dtype=object)
        if Phi_arr.shape != (self.D, self.D):
            raise ValueError(f"Phi must have shape ({self.D}, {self.D}).")

        is_callable = np.array(
            [[callable(Phi_arr[i, j]) for j in range(self.D)] for i in range(self.D)]
        )

        if is_callable.all():
            self._mode = "custom"
            self.Phi = Phi_arr
            self.alpha = None
        elif (~is_callable).all():
            self._mode = "parametric"
            self.alpha = Phi_arr.astype(float)
            if np.any(self.alpha < 0):
                raise ValueError("alpha must be nonnegative in parametric mode.")
            self.Phi = np.empty((self.D, self.D), dtype=object)
            for i in range(self.D):
                for j in range(self.D):
                    a, w = float(self.alpha[i, j]), self.omega
                    self.Phi[i, j] = (lambda _a, _w: (lambda t: _a * _w * np.exp(-_w * np.asarray(t))))(a, w)
        else:
            raise ValueError("Phi must be either all callables or all scalars.")
        
        if self._mode == "custom":
            cutoffs = []
            for i in range(self.D):
                for j in range(self.D):
                    if hasattr(self.Phi[i, j], "cutoff"):
                        cutoffs.append(float(self.Phi[i, j].cutoff))
            self.memory_window = max(cutoffs) if cutoffs else self.T_max
        else:
            self.memory_window = np.inf

    def check_stability(self, n_grid=5000):
        """
        Compute spectral radius $\rho(||\Phi||)$, where ||\Phi||_{ij} = ∫ \phi^{ij}(t) dt.
        """
        t_grid = np.linspace(1e-8, self.T_max, n_grid)
        norm_Phi = np.zeros((self.D, self.D), dtype=float)

        for i in range(self.D):
            for j in range(self.D):
                vals = np.asarray(self.Phi[i, j](t_grid), dtype=float)
                norm_Phi[i, j] = np.trapezoid(vals, t_grid)

        eigvals = np.linalg.eigvals(norm_Phi)
        rho = float(np.max(np.abs(eigvals)))
        print(f"Max eigenvalue {rho}")
        if rho < 1:
            print("Stable")
        else:
            print("Unstable or critical")
        return rho

    def kernel_norm_matrix(self, n_grid=5000):
        """
        Return K_{ij} = ∫ \phi^{ij}(t) dt over [0, T_max].
        """
        t_grid = np.linspace(1e-8, self.T_max, n_grid)
        K = np.zeros((self.D, self.D), dtype=float)
        for i in range(self.D):
            for j in range(self.D):
                vals = np.asarray(self.Phi[i, j](t_grid), dtype=float)
                K[i, j] = np.trapezoid(vals, t_grid)
        return K

    def stationary_rate(self, n_grid=5000):
        """
        Return $\Lambda = (I - K)^(-1) \mu$ for custom kernels, or equivalently for parametric.
        """
        K = self.kernel_norm_matrix(n_grid=n_grid)
        return np.linalg.solve(np.eye(self.D) - K, self.mu)

    def lambda_i(self, t, data, i):
        if len(data) == 0:
            return float(self.mu[i])

        past = np.asarray(data, dtype=float)
        past = past[past[:, 0] < t]

        if self._mode == "custom" and np.isfinite(self.memory_window):
            past = past[past[:, 0] >= t - self.memory_window]

        if len(past) == 0:
            return float(self.mu[i])

        val = self.mu[i]
        for tk, j in past:
            val += self.Phi[i, int(j)](t - tk)
        return float(val)

    def lambda_t(self, t, data):
        """
        Return the full intensity vector.
        """
        return np.array([self.lambda_i(t, data, i) for i in range(self.D)], dtype=float)

    def generate(self, horizon=10.0, seed=None):
        """
        Simulate the process on [0, horizon] via thinning.
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        rng = np.random.default_rng(seed)

        if self._mode == "parametric":
            return self._generate_parametric(horizon=horizon, rng=rng)
        else:
            return self._generate_custom_safe(horizon=horizon, rng=rng)

    def _generate_parametric(self, horizon, rng):
        """
        Exact Ogata thinning update for exponential kernels:
            \phi^{ij}(t) = \alpha^{ij} \omega \exp(-\omega t)
        """
        Istar = float(np.sum(self.mu))
        if Istar <= 0:
            return np.empty((0, 2), dtype=float)

        s = rng.exponential(1.0 / Istar)
        if s > horizon:
            return np.empty((0, 2), dtype=float)

        j0 = int(rng.choice(self.D, p=self.mu / Istar))
        data = np.array([[s, float(j0)]], dtype=float)

        lastrates = self.mu.copy()
        decIstar = False

        while True:
            tj, uj = data[-1, 0], int(data[-1, 1])

            if decIstar:
                Istar = float(np.sum(rates))
                decIstar = False
            else:
                Istar = float(np.sum(lastrates) + self.omega * np.sum(self.alpha[:, uj]))

            s = s + rng.exponential(1.0 / Istar)
            if s > horizon:
                return data

            rates = self.mu + np.exp(-self.omega * (s - tj)) * (
                self.alpha[:, uj].flatten() * self.omega + lastrates - self.mu
            )

            diff = max(Istar - float(np.sum(rates)), 0.0)
            probs = np.append(rates, diff) / Istar
            n0 = int(rng.choice(self.D + 1, p=probs))

            if n0 < self.D:
                data = np.append(data, [[s, float(n0)]], axis=0)
                lastrates = rates.copy()
            else:
                decIstar = True

    def _kernel_sup_matrix(self, n_grid=5000):
        """
        S_{ij} = sup_t \phi^{ij}(t) over [0, T_max].
        This gives a valid dominating bound for arbitrary nonnegative kernels.
        """
        t_grid = np.linspace(0.0, self.T_max, n_grid)
        S = np.zeros((self.D, self.D), dtype=float)
        for i in range(self.D):
            for j in range(self.D):
                vals = np.asarray(self.Phi[i, j](t_grid), dtype=float)
                S[i, j] = float(np.max(vals))
        return S

    def _generate_custom_safe(self, horizon, rng):
        if np.sum(self.mu) <= 0:
            return np.empty((0, 2), dtype=float)

        S = self._kernel_sup_matrix()
        jump_bound_by_source = S.sum(axis=0)

        times = []
        types = []
        t = 0.0

        while True:
            if len(types) == 0:
                Istar = float(np.sum(self.mu))
                recent_idx = np.array([], dtype=int)
            else:
                times_arr = np.asarray(times, dtype=float)
                types_arr = np.asarray(types, dtype=int)

                recent_idx = np.where(times_arr >= t - self.memory_window)[0]
                recent_types = types_arr[recent_idx]

                counts = np.bincount(recent_types, minlength=self.D)
                Istar = float(np.sum(self.mu) + np.dot(jump_bound_by_source, counts))

            if Istar <= 0:
                break

            t = t + rng.exponential(1.0 / Istar)
            if t > horizon:
                break

            if len(times) == 0:
                data = np.empty((0, 2), dtype=float)
            else:
                times_arr = np.asarray(times, dtype=float)
                types_arr = np.asarray(types, dtype=float)

                if np.isfinite(self.memory_window):
                    mask = times_arr >= t - self.memory_window
                    data = np.column_stack([times_arr[mask], types_arr[mask]])
                else:
                    data = np.column_stack([times_arr, types_arr])

            rates = self.lambda_t(t, data)
            total = float(np.sum(rates))

            if total > Istar + 1e-12:
                raise RuntimeError(
                    f"Invalid upper bound in custom thinning: total={total:.6f} > Istar={Istar:.6f}."
                )

            if total > 0 and rng.uniform() <= total / Istar:
                event_type = int(rng.choice(self.D, p=rates / total))
                times.append(t)
                types.append(event_type)

        if len(times) == 0:
            return np.empty((0, 2), dtype=float)

        return np.column_stack([np.asarray(times, dtype=float), np.asarray(types, dtype=float)])

    def plot_intensity(self, data, horizon=10.0, n_points=500):
        """
        Plot \lambda^i_t for all components i over [0, horizon].
        Event times are shown as tick marks on the time axis.
        """
        t_grid = np.linspace(1e-4, horizon, n_points)
        fig, axes = plt.subplots(self.D, 1, figsize=(10, 2.5 * self.D), sharex=True)
        if self.D == 1:
            axes = [axes]

        cmap = plt.cm.tab10(np.linspace(0, 0.6, self.D))
        data = np.asarray(data, dtype=float)

        for i, ax in enumerate(axes):
            lam = [self.lambda_i(t, data, i) for t in t_grid]
            evts = data[data[:, 1] == i, 0] if len(data) > 0 else np.array([])
            ax.plot(t_grid, lam, color=cmap[i], lw=1.5, label=f"$\\lambda^{{{i+1}}}_t$")
            ax.scatter(evts, np.zeros_like(evts), marker="|", s=80, color=cmap[i], alpha=0.7, zorder=3)
            ax.set_ylabel(f"$\\lambda^{{{i+1}}}_t$", fontsize=13)
            ax.legend(loc="upper right", fontsize=11)
            ax.set_xlim(0, horizon)

        axes[-1].set_xlabel("$t$", fontsize=13)
        plt.suptitle("Conditional intensities $\\lambda^i_t$", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_kernels(self, t_max=5.0, n_points=300):
        """
        Plot all kernel functions $\phi^{ij}(t)$ on [0, t_max].
        """
        t_grid = np.linspace(1e-6, t_max, n_points)
        fig, axes = plt.subplots(
            self.D,
            self.D,
            figsize=(3.5 * self.D, 3.0 * self.D),
            sharex=True,
            sharey=False,
        )
        if self.D == 1:
            axes = np.array([[axes]])

        for i in range(self.D):
            for j in range(self.D):
                ax = axes[i, j]
                phi = np.asarray(self.Phi[i, j](t_grid), dtype=float)
                ax.plot(t_grid, phi, lw=1.5, color="#378ADD")
                ax.fill_between(t_grid, phi, alpha=0.12, color="#378ADD")
                ax.set_title(f"$\\varphi^{{{i+1}{j+1}}}(t)$", fontsize=12)
                ax.set_xlim(0, t_max)
                ax.set_ylim(bottom=0)
                if i == self.D - 1:
                    ax.set_xlabel("$t$", fontsize=11)
                if j == 0:
                    ax.set_ylabel("kernel value", fontsize=10)

        plt.suptitle("Kernel matrix $\\Phi = (\\varphi^{ij})$", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()


def nn_to_phi(models, t_scaler, x_scaler, mark_grid=None, mark_weights=None):
    """
    Convert D trained models into a (D×D) object array of callables ready for MHP(Phi=...).
    """
    D = len(models)

    if mark_grid is not None:
        mark_grid = np.asarray(mark_grid, dtype=float)
        mark_weights = np.asarray(mark_weights, dtype=float)
        assert len(mark_grid) == len(mark_weights), "mark_grid and mark_weights must have the same length."
        assert np.isclose(mark_weights.sum(), 1.0), "mark_weights must sum to 1."
    else:
        mark_grid = np.array([1.0])
        mark_weights = np.array([1.0])

    Phi = np.empty((D, D), dtype=object)

    for i in range(D):
        for j in range(D):
            def _make(model_i, col_j, t_sc, x_sc, mg, mw):
                def phi_ij(t, x=None):
                    t = np.atleast_1d(np.asarray(t, dtype=float))

                    if x is None:
                        t_rep = np.repeat(t, len(mg))
                        x_rep = np.tile(mg, len(t))
                        out = model_i.predict(t_sc(t_rep), x_sc(x_rep))
                        vals = out[:, col_j].reshape(len(t), len(mg))
                        result = (vals * mw).sum(axis=1)
                    else:
                        x = np.broadcast_to(np.asarray(x, dtype=float), t.shape)
                        out = model_i.predict(t_sc(t), x_sc(x))
                        result = out[:, col_j]

                    return result.squeeze()

                return phi_ij

            Phi[i, j] = _make(models[i], j, t_scaler, x_scaler, mark_grid, mark_weights)

    return Phi