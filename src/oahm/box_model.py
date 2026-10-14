"""Нулевая (боксовая) модель теплового взаимодействия океана и атмосферы.

Система связана тремя резервуарами с температурами:

- Ta -- приземная атмосфера (теплоёмкость ca);
- Tm -- верхний квазиоднородный слой (ВКС) океана (теплоёмкость cm);
- Td -- глубинный слой океана (теплоёмкость cd).

Уравнения теплового баланса (время в годах, теплоёмкости -- отн. ед.):

    dTa/dt = ( -lam*(Ta - Tm) - la*Ta + dq ) / ca
    dTm/dt = ( -lam*(Tm - Ta) - lmd*(Tm - Td) ) / cm
    dTd/dt = ( -lmd*(Td - Tm) ) / cd

где lam  -- интенсивность теплообмена атмосфера <-> ВКС;
     lmd -- интенсивность теплообмена ВКС <-> глубинный слой;
     la  -- радиационное выхолаживание атмосферы;
     dq  -- внешний приток тепла к атмосфере.

Стационарное решение полной системы: Ta = Tm = Td = dq / la.
Система линейная и устойчивая (все собственные числа матрицы < 0), поэтому
интегрирование до установления служит проверкой корректности схемы.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class BoxParams:
    """Параметры боксовой модели; нулевой коэффициент связи «выключает»
    соответствующий механизм обмена (эксперименты 2 и 3)."""

    ca: float = 0.45   # теплоёмкость атмосферы
    cm: float = 10.0   # теплоёмкость ВКС
    cd: float = 100.0  # теплоёмкость глубинного слоя
    lam: float = 45.0  # обмен атмосфера <-> ВКС
    lmd: float = 2.0   # обмен ВКС <-> глубинный слой
    la: float = 2.4    # радиационное выхолаживание атмосферы
    dq: float = 1.0    # внешний приток тепла

    @property
    def steady_state(self) -> np.ndarray:
        """Аналитическое стационарное решение [Ta, Tm, Td] при нулевых
        начальных условиях (как в численных экспериментах).

        Полная связь: все резервуары выравниваются на dq/la. При lam = 0
        атмосфера изолирована, океан остывает некуда -> Tm = Td = 0.
        При lmd = 0 глубинный слой изолирован -> Td = 0, атмосфера и ВКС
        обмениваются до dq/la.
        """
        t_eq = self.dq / self.la
        if self.lam == 0.0:
            return np.array([t_eq, 0.0, 0.0])
        if self.lmd == 0.0:
            return np.array([t_eq, t_eq, 0.0])
        return np.full(3, t_eq)


def rhs(state: np.ndarray, p: BoxParams) -> np.ndarray:
    """Правая часть системы; state = [Ta, Tm, Td]."""
    Ta, Tm, Td = state
    return np.array([
        (-p.lam * (Ta - Tm) - p.la * Ta + p.dq) / p.ca,
        (-p.lam * (Tm - Ta) - p.lmd * (Tm - Td)) / p.cm,
        (-p.lmd * (Td - Tm)) / p.cd,
    ])


def integrate(
    p: BoxParams,
    years: float = 30.0,
    dt: float = 1.0 / 365.0,
    state0: np.ndarray | None = None,
    scheme: str = "rk4",
) -> tuple[np.ndarray, np.ndarray]:
    """Проинтегрировать систему на сетке dt в течение `years` лет.

    scheme: "rk4" (классический Рунге--Кутты 4-го порядка) или "euler".
    Возвращает (t, states), t формы (n+1,), states формы (n+1, 3).
    """
    if scheme not in ("rk4", "euler"):
        raise ValueError(f"Неизвестная схема: {scheme!r}")

    n = int(np.round(years / dt))
    t = np.arange(n + 1) * dt
    states = np.empty((n + 1, 3))
    states[0] = np.zeros(3) if state0 is None else np.asarray(state0, float)

    for i in range(n):
        s = states[i]
        if scheme == "euler":
            states[i + 1] = s + dt * rhs(s, p)
        else:
            k1 = rhs(s, p)
            k2 = rhs(s + 0.5 * dt * k1, p)
            k3 = rhs(s + 0.5 * dt * k2, p)
            k4 = rhs(s + dt * k3, p)
            states[i + 1] = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, states


def relaxation_timescale(p: BoxParams) -> float:
    """Время установления (e-фолдинг) самого медленного процесса системы:
    1 / |lambda_min| матрицы оператора, годы.

    У полной модели это глубокий океан: cd/lmd = 50 лет, поэтому выход в
    равновесие занимает несколько сотен лет.
    """
    A = operator(p)
    eig = np.linalg.eigvals(A)
    return 1.0 / np.abs(eig).min()


def operator(p: BoxParams) -> np.ndarray:
    """Матрица линеаризованной системы в отклонениях от стационара."""
    return np.array([
        [(-p.lam - p.la) / p.ca, p.lam / p.ca, 0.0],
        [p.lam / p.cm, (-p.lam - p.lmd) / p.cm, p.lmd / p.cm],
        [0.0, p.lmd / p.cd, -p.lmd / p.cd],
    ])


def analytic_solution(p: BoxParams, t: np.ndarray,
                      state0: np.ndarray | None = None) -> np.ndarray:
    """Точное решение линейной системы через спектральное разложение.

    state(t) = x_eq + V @ diag(exp(lam_i * t)) @ V^{-1} @ (state0 - x_eq).
    Форма результата: (len(t), 3). Используется как эталон при проверке
    фактического порядка точности численной схемы.
    """
    x0 = np.zeros(3) if state0 is None else np.asarray(state0, float)
    A = operator(p)
    x_eq = _steady_for(p, x0)

    eigvals, V = np.linalg.eig(A)
    dev = np.linalg.solve(V, (x0 - x_eq))
    t = np.asarray(t, float)
    modes = np.exp(np.outer(t, eigvals)) * dev  # (len(t), 3)
    return x_eq + np.real(modes @ V.T)


def _steady_for(p: BoxParams, state0: np.ndarray) -> np.ndarray:
    """Стационар, достижимый из начального состояния state0.

    Атмосфера всегда имеет внешний приток q и радиационный сток, поэтому
    её равновесие равно q/la независимо от связи с океаном; изолированные
    океанские резервуары сохраняют тепло. При lmd = 0 ВКС следует за
    атмосферой, а глубинный слой остаётся при начальной температуре.
    """
    t_eq = p.dq / p.la
    out = np.empty(3)
    out[0] = t_eq
    if p.lmd != 0.0:
        # при lam = 0 океанские боксы изолированы: сохраняется cm*Tm + cd*Td
        shared = t_eq if p.lam != 0.0 else \
            (p.cm * state0[1] + p.cd * state0[2]) / (p.cm + p.cd)
        out[1] = out[2] = shared
    else:
        out[1] = t_eq if p.lam != 0.0 else state0[1]
        out[2] = state0[2]
    return out


def convergence_error(p: BoxParams, dts: np.ndarray, t_final: float = 10.0,
                      scheme: str = "rk4") -> np.ndarray:
    """Ошибка численного решения в момент t_final относительно точного.

    Для каждого dt из dts возвращает норму отклонения от аналитического
    решения; наклон log2(err) по log2(dt) -- фактический порядок схемы.
    """
    exact = analytic_solution(p, np.array([t_final]))[0]
    err = np.empty(len(dts))
    for j, dt in enumerate(dts):
        _, states = integrate(p, years=t_final, dt=dt, scheme=scheme)
        err[j] = np.linalg.norm(states[-1] - exact)
    return err


def with_overrides(p: BoxParams, **overrides) -> BoxParams:
    """Копия параметров с заменой отдельных коэффициентов (удобно для
    численных экспериментов: with_overrides(p, lam=0.0))."""
    return replace(p, **overrides)
