"""Линейно-резервуарная модель трансформации осадков в речной сток.

Модель "линейного резервуара" (базовая ячейка модели Калинина--Милюкова):
бассейн с осадками сопоставляется резервуару, приток которого пропорционален
слою выпавших осадков, а сток -- запасу воды:

    tau * dQ/dt = a * R(t) - Q(t),

где Q -- расход воды, м3/с; R -- слой осадков, м/сут;
    a   -- эмпирический коэффициент (эффективная площадь бассейна),
           м3/с на (м/сут);
    tau -- время релаксации резервуара, сут.

В лабораторной работе модель интегрируется тремя разностными схемами
с шагом dt = 1 сут:

- явный Эйлер (forward):   Q_{n+1} = Q_n + dt*(a*R_n - Q_n)/tau;
- неявный Эйлер (backward): Q_{n+1}*(1 + dt/tau) = Q_n + dt*a*R_n/tau
  (приток R_n известен на момент выпуска прогноза);
- чехарда (leapfrog):       Q_{n+1} = Q_{n-1} + 2*dt*(a*R_n - Q_n)/tau.

Калибровка параметров (a, tau) -- оптимизационная задача: минимизируется
RMSE прогноза расхода на обучающем периоде; качество проверяется на
отложенном периоде относительно наивного baseline'а (инерционный прогноз
Q_pred(t+1) = Q(t), ему соответствует S/sigma_dDelta = 1).

Верификационные метрики (принятые в гидрологических прогнозах):
- S  -- среднеквадратическая ошибка прогноза;
- sigma_dDelta -- СКО суточных изменений расхода (ошибка baseline'а);
- S/sigma_dDelta -- отношение; прогноз оправдан (имеет навык), если < 1;
- P, % -- доля оправдавшихся прогнозов (ошибка не превышает допустимую
  0.674*sigma_dDelta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCHEMES = ("forward", "backward", "leapfrog")


# ---------------------------------------------------------------- данные --

def load_discharge(path: str) -> pd.Series:
    """Расход воды Q, м3/с, из широкого Excel-файла гидропоста.

    Формат исходника: столбцы Year, Day и 12 столбцов-месяцев; каждая
    строка -- день месяца. Разворачивается в суточный ряд с индексом Date.
    """
    wide = pd.read_excel(path)
    wide.columns = [int(c) if str(c).replace(".0", "").isdigit() else str(c)
                    for c in wide.columns]
    long = wide.melt(id_vars=["Year", "Day"], var_name="Month",
                     value_name="Q").dropna(subset=["Q"])
    dates = pd.to_datetime(
        dict(year=long["Year"], month=long["Month"], day=long["Day"]),
        errors="coerce",
    )
    q = pd.Series(pd.to_numeric(long["Q"], errors="coerce").to_numpy(),
                  index=dates, name="Q").dropna()
    return q.sort_index()


def load_precipitation(path: str) -> pd.Series:
    """Слой осадков R, м/сут, из файла метеостанции (разделитель ';').

    Исходные значения в мм пересчитываются в метры (как в лабораторной).
    """
    raw = pd.read_csv(path, sep=";", header=None,
                      names=["station", "Year", "Month", "Day", "R_mm"],
                      usecols=["Year", "Month", "Day", "R_mm"])
    dates = pd.to_datetime(
        dict(year=raw["Year"], month=raw["Month"], day=raw["Day"]),
        errors="coerce",
    )
    r = pd.Series(pd.to_numeric(raw["R_mm"], errors="coerce").to_numpy() * 1e-3,
                  index=dates, name="R").dropna()
    return r.sort_index()


def make_dataset(discharge: pd.Series, precipitation: pd.Series) -> pd.DataFrame:
    """Объединить ряды по датам; вернуть колонки R (м/сут) и Q (м3/с)."""
    df = pd.concat([precipitation, discharge], axis=1, join="inner").dropna()
    df = df.sort_index()
    gaps = int((df.index.to_series().diff().dt.days > 1).sum())
    if gaps:
        print(f"Внимание: в суточном ряде {gaps} разрывов (дни пропущены)")
    return df


# ----------------------------------------------------------------- модель --

@dataclass(frozen=True)
class LinearReservoirParams:
    """Параметры линейного резервуара."""

    gain: float      # a: м3/с на (м/сут) осадков
    tau_days: float  # tau: время релаксации, сут


def simulate(Q0: float, R: np.ndarray, params: LinearReservoirParams,
             scheme: str = "forward", dt: float = 1.0) -> np.ndarray:
    """Прогноз расхода на сетке dt; возвращает ряд длины len(R) + 1.

    Q[0] = Q0 (наблюдение), далее пошаговый прогноз по выбранной схеме.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Схема должна быть одной из {SCHEMES}")

    R = np.asarray(R, dtype=float)
    Q = np.empty(R.size + 1)
    Q[0] = Q0

    a, tau = params.gain, params.tau_days
    if scheme == "forward":
        for i in range(R.size):
            Q[i + 1] = Q[i] + dt * (a * R[i] - Q[i]) / tau
    elif scheme == "backward":
        # Q_{i+1}*(1 + dt/tau) = Q_i + dt*a*R_i/tau
        for i in range(R.size):
            Q[i + 1] = (Q[i] + dt * a * R[i] / tau) / (1.0 + dt / tau)
    else:  # leapfrog: нужен старт -- один шаг явной схемы
        Q[1] = Q[0] + dt * (a * R[0] - Q[0]) / tau
        for i in range(1, R.size):
            Q[i + 1] = Q[i - 1] + 2 * dt * (a * R[i] - Q[i]) / tau
    return Q


# ------------------------------------------------------------- метрики --

def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Среднеквадратическая ошибка прогноза S."""
    err = np.asarray(obs, float) - np.asarray(pred, float)
    return float(np.sqrt(np.nanmean(err**2)))


def sigma_delta(obs: np.ndarray) -> float:
    """СКО суточных изменений расхода -- ошибка наивного инерционного
    прогноза Q_pred(t+1) = Q(t)."""
    return float(np.std(np.diff(np.asarray(obs, float)), ddof=1))


def verification(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Верификация прогноза относительно инерционного baseline'а.

    Возвращает S, sigma_dDelta, S/sigma_dDelta, навык 1 - S/sigma_dDelta
    и долю оправдавшихся прогнозов P (ошибка <= 0.674*sigma_dDelta).
    """
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    s = rmse(obs[mask], pred[mask])
    sd = sigma_delta(obs[mask])
    allowed = 0.674 * sd
    return {
        "S, м3/с": s,
        "sigma_dDelta, м3/с": sd,
        "S/sigma_dDelta": s / sd,
        "skill (1 - S/sigma_dDelta)": 1.0 - s / sd,
        "P, %": float(np.mean(np.abs(obs[mask] - pred[mask]) <= allowed) * 100),
    }


# ------------------------------------------------------------ калибровка --

@dataclass(frozen=True)
class CalibrationResult:
    """Результат калибровки: лучшие параметры, RMSE, сетка потерь."""

    params: LinearReservoirParams
    rmse: float
    grid_gains: np.ndarray
    grid_taus: np.ndarray
    loss: np.ndarray  # форма (len(grid_gains), len(grid_taus))


def calibrate(
    obs: np.ndarray,
    R: np.ndarray,
    Q0: float,
    scheme: str = "forward",
    gain_range: tuple[float, float] = (1e2, 1e5),
    tau_range: tuple[float, float] = (1.0, 200.0),
    grid_size: tuple[int, int] = (40, 40),
    refine: bool = True,
) -> CalibrationResult:
    """Калибровка (a, tau): минимизация RMSE на обучающем периоде.

    Этап 1 -- грубый поиск по лог-равномерной сетке по a и линейной по tau
    (возвращает и карту потерь для визуализации);
    этап 2 (refine) -- уточнение методом L-BFGS-B в лог-пространстве
    параметров в пределах тех же границ (scipy.optimize).
    """
    obs = np.asarray(obs, float)
    R = np.asarray(R, float)
    if obs.shape != R.shape:
        raise ValueError("obs и R должны быть рядами одной длины")
    n = obs.size
    gains = np.geomspace(*gain_range, grid_size[0])
    taus = np.linspace(*tau_range, grid_size[1])
    loss = np.full(grid_size, np.nan)

    def safe_rmse(params: LinearReservoirParams) -> float:
        # неустойчивые комбинации (например, leapfrog при больших dt/tau)
        # дают расходящиеся решения -- возвращаем для них большой штраф
        with np.errstate(over="ignore", invalid="ignore"):
            pred = simulate(Q0, R, params, scheme)
            s = rmse(obs[1:], pred[1:n])
        return s if np.isfinite(s) else 1e12

    for i, g in enumerate(gains):
        for j, tau in enumerate(taus):
            # pred[i] -- прогноз на день i; день 0 задан наблюдением
            loss[i, j] = safe_rmse(LinearReservoirParams(g, tau))

    i0, j0 = np.unravel_index(np.nanargmin(loss), loss.shape)
    best = LinearReservoirParams(float(gains[i0]), float(taus[j0]))
    best_loss = float(loss[i0, j0])

    if refine:
        def objective(theta):
            return safe_rmse(LinearReservoirParams(*np.exp(theta)))

        res = minimize(
            objective,
            x0=np.log([best.gain, best.tau_days]),
            method="L-BFGS-B",
            bounds=[np.log(gain_range), np.log(tau_range)],
        )
        if res.fun < best_loss:
            best = LinearReservoirParams(*np.exp(res.x))
            best_loss = float(res.fun)

    return CalibrationResult(best, best_loss, gains, taus, loss)
