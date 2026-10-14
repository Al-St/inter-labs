"""1-D модель длинных волн в бассейне постоянной глубины.

Линейные уравнения мелкой воды (без трения и вращения Земли):

    dQ/dt = -g*h * d(eta)/dx        -- уравнение движения
    d(eta)/dt = -dQ/dx              -- уравнение неразрывности

где eta -- отклонение уровня, Q -- расход через вертикальное сечение
единичной ширины, h -- глубина бассейна. Фазовая скорость волны
C = sqrt(g*h), длина волны lam = T*C для периода T.

Разностная схема на разнесённой (staggered) сетке C-типа: Q определён на
границах ячеек (N+1 узел), eta -- в центрах (N узлов); расщепление
"вперёд-назад" (forward-backward) имеет первый порядок точности и при
числе Куранта CFL = C*dt/dx <= 1 устойчиво:

    Q_j^{k+1} = Q_j^k - dt*g*h*(eta_right - eta_left)/dx
    eta_i^{k+1} = eta_i^k - dt*(Q_{i+1}^{k+1} - Q_i^{k+1})/dx

Граничные условия:
- твердая стенка:  Q = 0;
- открытая граница: внешний уровень eta_ext(t) задан, изнутри волна
  свободно проходит через границу.

Эксперименты лабораторной работы:
1. "входящая волна" -- на открытой границе задается один период
   синусоидального возмущения уровня; пакет проходит бассейн, отражается
   от твердой стенки;
2. "начальное возвышение" -- в замкнутом бассейне задается купол
   возвышения амплитуды A и длины lam/2; распад на две волны и сейшевые
   колебания.

Приближение "длинной волны" численно дисперсивно: фазовая скорость на
сетке зависит от k*dx (scheme dispersion), поэтому пакет со временем
деформируется; рост ошибки амплитуды |1 - max(eta)/A| отслеживается в
экспериментах.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

G = 9.81  # м/с^2


@dataclass(frozen=True)
class WaveExperiment:
    """Результат численного эксперимента.

    eta: (n_steps+1, N) -- возвышения уровня в центрах ячеек;
    Q:   (n_steps+1, N+1) -- расходы на границах ячеек;
    t:   (n_steps+1,) -- моменты времени, с;
    x:   (N,) -- координаты центров ячеек, м (0 -- твердая стенка).
    """

    eta: np.ndarray
    Q: np.ndarray
    t: np.ndarray
    x: np.ndarray
    L: float
    h: float
    A: float
    T: float
    dx: float
    dt: float
    mode: str

    @property
    def lam(self) -> float:
        """Длина волны T*sqrt(g*h), м."""
        return self.T * np.sqrt(G * self.h)

    @property
    def celerity(self) -> float:
        return np.sqrt(G * self.h)

    @property
    def cfl(self) -> float:
        return self.celerity * self.dt / self.dx

    @property
    def amplitude_error(self) -> np.ndarray:
        """Относительное отклонение max|eta| от заданной амплитуды A
        (мера вычислительной дисперсии) на каждом шаге по времени."""
        return np.abs(1.0 - np.abs(self.eta).max(axis=1) / self.A)


def simulate_incoming_wave(
    L: float,
    h: float,
    A: float,
    T: float,
    duration_s: float = 15 * 3600,
    cells_per_wavelength: int = 40,
    CFL: float = 0.5,
) -> WaveExperiment:
    """Пакет из одного периода волны входит через открытую границу x = L.

    Слева (x = 0) -- твердая стенка: волна отражается, и на финале
    интерференция падающей и отраженной волн.
    """
    lam = T * np.sqrt(G * h)
    dx = lam / cells_per_wavelength
    dt = CFL * dx / np.sqrt(G * h)
    n_steps = int(np.round(duration_s / dt))
    N = int(np.round(L / dx))

    def eta_ext(step: int) -> float:
        # внешний уровень на открытой границе: один период синусоиды
        tau = step * dt
        if tau <= T:
            return A * np.cos(2 * np.pi * tau / T - np.pi / 2)
        return 0.0

    eta, Q = _simulate(L, h, N, dx, dt, n_steps, eta_ext)
    return _make_experiment(eta, Q, L, h, A, T, dx, dt, "incoming_wave")


def simulate_initial_elevation(
    L: float,
    h: float,
    A: float,
    T: float,
    duration_s: float = 3.75 * 3600,
    cells_per_wavelength: int = 40,
    CFL: float = 0.5,
) -> WaveExperiment:
    """Купол возвышения (полуволна cos длиной lam/2, амплитуда A) в центре
    замкнутого бассейна: распад на две волны и сейшевые колебания."""
    lam = T * np.sqrt(G * h)
    dx = lam / cells_per_wavelength
    dt = CFL * dx / np.sqrt(G * h)
    n_steps = int(np.round(duration_s / dt))
    N = int(np.round(L / dx))

    eta0 = np.zeros(N)
    # ширина купола lam/2, максимум A в центре; на краях cos(-pi/2) = 0
    half = cells_per_wavelength // 2
    i_left = N // 2 - half // 2
    idx = np.arange(i_left, i_left + half + 1)
    eta0[idx] = A * np.cos(2 * np.pi * (idx - i_left) * dx / lam - np.pi / 2)

    eta, Q = _simulate(L, h, N, dx, dt, n_steps, None, eta0=eta0)
    return _make_experiment(eta, Q, L, h, A, T, dx, dt, "initial_elevation")


def _simulate(
    L: float,
    h: float,
    N: int,
    dx: float,
    dt: float,
    n_steps: int,
    eta_ext=None,
    eta0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Основной решатель: разнесенная сетка, расщепление вперед-назад.

    eta_ext(step) -- уровень вне открытой границы; None -- обе границы
    твердые (замкнутый бассейн).
    """
    eta = np.zeros((n_steps + 1, N))
    Q = np.zeros((n_steps + 1, N + 1))
    eta[0] = 0.0 if eta0 is None else eta0

    gh_dt_dx = G * h * dt / dx
    dt_dx = dt / dx
    open_boundary = eta_ext is not None

    for k in range(n_steps):
        # --- движение: Q на внутренних гранях ---
        Q[k + 1, 1:N] = Q[k, 1:N] - gh_dt_dx * (eta[k, 1:] - eta[k, :-1])
        if open_boundary:
            Q[k + 1, N] = Q[k, N] - gh_dt_dx * (eta_ext(k) - eta[k, N - 1])
        # Q[0] = Q[N] = 0 -- твердые стенки

        # --- неразрывность: eta в центрах ячеек ---
        eta[k + 1] = eta[k] - dt_dx * (Q[k + 1, 1:] - Q[k + 1, :-1])

    return eta, Q


def _make_experiment(eta, Q, L, h, A, T, dx, dt, mode) -> WaveExperiment:
    n_steps = eta.shape[0] - 1
    N = eta.shape[1]
    t = np.arange(n_steps + 1) * dt
    x = (np.arange(N) + 0.5) * dx
    return WaveExperiment(eta=eta, Q=Q, t=t, x=x, L=L, h=h, A=A, T=T,
                          dx=dx, dt=dt, mode=mode)


def snapshot_plot(exp: WaveExperiment, steps: list[int], path: str | None = None,
                  ylim: float | None = None, title: str = ""):
    """Профили уровня в выбранные моменты времени (мгновенные снимки)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(steps)))
    for color, k in zip(colors, steps):
        ax.plot(exp.x / 1000, exp.eta[k], lw=1.6, color=color,
                label=f"t = {exp.t[k] / 3600:.1f} ч")
    ax.axhline(exp.A, color="grey", ls="--", lw=0.8, label="±A")
    ax.axhline(-exp.A, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Расстояние от твердой стенки, км")
    ax.set_ylabel("Отклонение уровня, м")
    ax.set_title(title)
    ax.set_ylim(-(ylim or 1.6 * exp.A), (ylim or 1.6 * exp.A))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig, ax


def animate(exp: WaveExperiment, path: str, fps: int = 15,
            n_frames: int = 150, title: str = "") -> str:
    """Сохранить GIF-анимацию распространения волны (PillowWriter)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    stride = max(1, exp.eta.shape[0] // n_frames)
    steps = np.arange(0, exp.eta.shape[0], stride)

    (line,) = ax.plot(exp.x / 1000, exp.eta[0], lw=2)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12)

    def update(k):
        line.set_ydata(exp.eta[k])
        time_text.set_text(f"t = {exp.t[k] / 3600:.1f} ч")
        return line, time_text

    ylim = 1.6 * exp.A
    ax.set_ylim(-ylim, ylim)
    ax.set_xlim(0, exp.L / 1000)
    ax.set_xlabel("Расстояние от твердой стенки, км")
    ax.set_ylabel("Отклонение уровня, м")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    anim = FuncAnimation(fig, update, frames=steps, blit=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
