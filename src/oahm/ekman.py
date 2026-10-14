"""Стационарный пограничный слой Экмана: атмосфера над океаном и дрейфовое
течение в верхнем слое океана.

Задача о годографе скорости при прохождении через границу раздела
океан -- атмосфера (динамически связанный пограничный слой):

- в атмосфере (высота z >= 0) геострофический ветер Vg на верхней границе
  слоя раскручивается к поверхности с поворотом влево (северное полушарие):

      u(z) = Vg - (Vg - V0) * exp(-A_a z) * cos(A_a z)
      v(z) =          (Vg - V0) * exp(-A_a z) * sin(A_a z)

- в океане (глубина z >= 0) дрейфовое течение затухает с глубиной
  с поворотом вправо (спираль Экмана):

      u(z) = V0 * exp(-A_w z) * cos(A_w z)
      v(z) = -V0 * exp(-A_w z) * sin(A_w z)

где A_i = sqrt(omega_z / k_i), omega_z = Omega*sin(phi) -- вертикальная
составляющая вектора угловой скорости вращения Земли (в обозначениях
задания), k_i -- коэффициент турбулентной вязкости.

Скорость V0 на границе раздела находится из непрерывности тангенциального
напряжения трения (энергия/импульс передаётся без скачка):

    eps = (rho_a / rho_w) * sqrt(k_a / k_w),    V0 = eps / (1 + eps) * Vg.

Толщины слоёв (глубины проникновения):  H_i = pi * sqrt(k_i / omega_z).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OMEGA = 7.2921158553e-5  # угловая скорость вращения Земли, рад/с


@dataclass(frozen=True)
class EkmanSolution:
    """Профили скорости в связанном пограничном слое атмосфера--океан.

    Атрибуты u_*/v_*/z_* — одноимённые массивы одинаковой длины;
    z_atm — высота над поверхностью, z_oc — глубина (обе положительные).
    """

    rho_w: float
    rho_a: float
    k_w: float
    k_a: float
    Vg: float
    phi_deg: float

    omega_z: float           # Omega*sin(phi), рад/с
    A_a: float               # обратный масштаб слоя в атмосфере, 1/м
    A_w: float               # обратный масштаб слоя в океане, 1/м
    H_atm: float             # толщина пограничного слоя атмосферы, м
    H_oc: float              # толщина верхнего слоя океана, м
    eps: float               # параметр связи rho_a/rho_w * sqrt(ka/kw)
    V0: float                # скорость на границе раздела, м/с

    z_atm: np.ndarray        # высоты, м
    u_atm: np.ndarray        # зональная компонента, м/с
    v_atm: np.ndarray        # меридиональная компонента, м/с

    z_oc: np.ndarray         # глубины, м
    u_oc: np.ndarray
    v_oc: np.ndarray

    @property
    def speed_atm(self) -> np.ndarray:
        return np.hypot(self.u_atm, self.v_atm)

    @property
    def speed_oc(self) -> np.ndarray:
        return np.hypot(self.u_oc, self.v_oc)


def solve_ekman(
    rho_w: float = 1000.0,
    rho_a: float = 1.3,
    k_w: float = 0.06,
    k_a: float = 14.0,
    Vg: float = 12.0,
    phi_deg: float = 65.0,
    dz_atm: float = 10.0,
    dz_oc: float = 1.0,
) -> EkmanSolution:
    """Рассчитать профили скорости в атмосфере и океане (спираль Экмана).

    Профили вычисляются в пределах слоёв [0, H_atm] и [0, H_oc] с шагами
    dz_atm и dz_oc; толщины слоёв H_i = pi*sqrt(k_i/f) округляются до
    целого числа шагов.
    """
    omega_z = OMEGA * np.sin(np.deg2rad(phi_deg))

    A_a = np.sqrt(omega_z / k_a)
    A_w = np.sqrt(omega_z / k_w)

    H_atm = np.pi * np.sqrt(k_a / omega_z)
    H_oc = np.pi * np.sqrt(k_w / omega_z)

    eps = rho_a / rho_w * np.sqrt(k_a / k_w)
    V0 = eps / (1.0 + eps) * Vg

    z_a = np.arange(int(np.round(H_atm / dz_atm)) + 1) * dz_atm
    z_w = np.arange(int(np.round(H_oc / dz_oc)) + 1) * dz_oc

    u_a = Vg - (Vg - V0) * np.exp(-A_a * z_a) * np.cos(A_a * z_a)
    v_a = (Vg - V0) * np.exp(-A_a * z_a) * np.sin(A_a * z_a)

    u_w = V0 * np.exp(-A_w * z_w) * np.cos(A_w * z_w)
    v_w = -V0 * np.exp(-A_w * z_w) * np.sin(A_w * z_w)

    return EkmanSolution(
        rho_w=rho_w, rho_a=rho_a, k_w=k_w, k_a=k_a, Vg=Vg, phi_deg=phi_deg,
        omega_z=omega_z, A_a=A_a, A_w=A_w, H_atm=H_atm, H_oc=H_oc, eps=eps, V0=V0,
        z_atm=z_a, u_atm=u_a, v_atm=v_a,
        z_oc=z_w, u_oc=u_w, v_oc=v_w,
    )


def hodograph_points(sol: EkmanSolution) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Годограф (u, v) вдоль пути атмосфера -> граница раздела -> океан.

    Возвращает (u, v, z_signed), где z_signed > 0 — высота в атмосфере,
    z_signed < 0 — глубина в океане. Атмосферная часть перевёрнута так,
    что кривая идёт от свободной атмосферы к поверхности.
    """
    u = np.concatenate([sol.u_atm[::-1], sol.u_oc[1:]])
    v = np.concatenate([sol.v_atm[::-1], sol.v_oc[1:]])
    z = np.concatenate([sol.z_atm[::-1], -sol.z_oc[1:]])
    return u, v, z
