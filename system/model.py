from typing import Dict

import numpy as np

from dataclasses import dataclass

dT = 0.0001

L = 5 / 1000  # vad ska det vara här? Läste 28 mH nånstans
B = 0.1  # 0 friktion beroende på varvtal
k_lambda = 0.2
R = 0.5 * 4

g = 9.82
m = 10
utväxlingskonstant = 9
verkningsgrad = 0.3
J = 0.00024  # moment konstant för motorn + system???


@dataclass
class SimulationResult:
    current: np.ndarray
    omega: np.ndarray

    time: np.ndarray

    voltages: Dict[str, np.ndarray]


def simulate(seconds: int, target_current: int) -> SimulationResult:
    N = int(seconds / dT)

    I = np.zeros(N)
    omega = np.zeros(N)

    V = np.zeros(N)

    vL = np.zeros(N)
    vR = np.zeros(N)
    vEa = np.zeros(N)

    for i in range(1, N):
        V[i] = 24 if I[i - 1] < target_current else 0

        vR[i] = R * I[i - 1]
        vEa[i] = k_lambda * omega[i - 1]
        vL[i] = V[i] - vR[i] - vEa[i]

        dI = (V[i] - R * I[i - 1] - k_lambda * omega[i - 1]) / L

        I[i] = I[i - 1] + dT * dI

        applied_force = m * (g + 0)
        T_last_min, T_last_max = screw_torque_with_friction(omega[i - 1], applied_force)

        T_last_min, T_last_max = T_last_min / utväxlingskonstant, T_last_max / utväxlingskonstant

        T_dev = k_lambda * I[i - 1]

        is_dynamic = T_last_min <= T_dev <= T_last_max

        if is_dynamic:
            dOmega = (T_dev - (B * omega[i - 1] + m * g / utväxlingskonstant / verkningsgrad)) / J
        else:
            dOmega = 0

        # print(dOmega)

        omega[i] = omega[i - 1] + dT * dOmega

    t = np.linspace(0, N * dT, N)

    return SimulationResult(
        current=I,
        omega=omega,
        time=t,
        voltages={
            "V_L": vL,
            "V_R": vR,
            "V_Ea": vEa,
            "v_Supply": V,
        }
    )


def screw_torque_with_friction(w, F):  # [Tmax,T,Tmin]
    screw_outer_dia = 0.025
    screw_inner_dia = 0.020
    pitch = 0.01  # m/revulution
    friction_coef_static = 0.15
    friction_coef_dynamic = 0.10
    no_load_friction_torque_static = 0.03
    no_load_friction_torque_dynamic = 0.02

    w_min_dynamic = 1  # rad/s speeds below this is considered standstill, shold be set to the maximum change during a simulation step.

    pitch_rad = pitch / 2 / np.pi  # mm/rad
    T_no_friction = F * pitch_rad

    R_eff = (screw_outer_dia + screw_inner_dia) / 4  # screw effective radius

    if abs(w) > w_min_dynamic:
        # dynamic friction
        T = T_no_friction + (abs(F) * R_eff * friction_coef_dynamic + no_load_friction_torque_dynamic) * np.sign(w)
        Tmax = T
        Tmin = T
    else:
        # static friction
        Tmax = T_no_friction + (abs(F) * R_eff * friction_coef_static + no_load_friction_torque_static)
        Tmin = T_no_friction - (abs(F) * R_eff * friction_coef_static + no_load_friction_torque_static)
    return Tmax, Tmin
