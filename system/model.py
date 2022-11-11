from typing import Dict

import numpy as np

from dataclasses import dataclass

dT = 0.0001

L = 5 / 1000  # vad ska det vara här? Läste 28 mH nånstans
B = 0.1  # 0 friktion beroende på varvtal
k_lambda = 0.2
R = 0.5

g = 9.82
m = 20
utväxlingskonstant = 9
verkningsgrad = 0.3
J = 0.00024  # moment konstant för motorn + system???


@dataclass
class SimulationResult:
    current: np.ndarray
    omega: np.ndarray
    pos: np.ndarray

    time: np.ndarray

    t_dev: np.ndarray
    t_last: np.ndarray

    voltages: Dict[str, np.ndarray]


def simulate(seconds: int, target_current: int) -> SimulationResult:
    N = int(seconds / dT)

    # Tillståndsvariabler
    I = np.zeros(N)
    omega = np.zeros(N)
    pos = np.zeros(N)

    V = np.zeros(N)

    vL = np.zeros(N)
    vR = np.zeros(N)
    vEa = np.zeros(N)

    T_dev = np.zeros(N)
    T_last = np.zeros(N)

    for i in range(1, N):
        V[i] = 24 if I[i - 1] < target_current else 0

        vR[i] = R * I[i - 1]
        vEa[i] = k_lambda * omega[i - 1]
        vL[i] = V[i] - vR[i] - vEa[i]

        dI = (V[i] - R * I[i - 1] - k_lambda * omega[i - 1]) / L

        I[i] = I[i - 1] + dT * dI

        applied_force = m * (g + 0)
        # T_last_max, T_last_min = screw_torque_with_friction(omega[i - 1], applied_force)

        # T_last_min, T_last_max = T_last_min / utväxlingskonstant, T_last_max / utväxlingskonstant

        T_dev[i] = k_lambda * I[i - 1]

        is_dynamic = abs(omega[i - 1]) > 1
        if is_dynamic:
            # dynamisk friction
            T_last[i] = screw_torque_with__dynamic_friction(omega[i - 1], applied_force)

        else:
            # static friction
            T_last_max, T_last_min = screw_torque_with_static_friction(applied_force)

            if T_last_min <= T_dev[i] <= T_last_max:
                T_last[i] = T_dev[i]
            else:
                if T_dev[i] > T_last_max:
                    T_last[i] = T_last_max
                else:
                    T_last[i] = T_last_min

        T_last[i] = T_last[i]# / utväxlingskonstant

        dOmega = (T_dev[i] - T_last[i]) / J

        omega[i] = omega[i - 1] + dT * dOmega

        dpos = omega[i] / 1800
        pos[i] = pos[i - 1] + dT * dpos

    t = np.linspace(0, N * dT, N)
    return SimulationResult(
        current=I,
        omega=omega,
        time=t,
        t_dev=T_dev,
        t_last=T_last,
        pos=pos,
        voltages={
            "V_L": vL,
            "V_R": vR,
            "V_Ea": vEa,
            "v_Supply": V,
        }
    )

screw_outer_dia = 0.024
screw_inner_dia = 0.019
pitch = 0.005  # m/revulution

friction_coef_static = 0.15
friction_coef_dynamic = 0.10

no_load_friction_torque_static = 0.03
no_load_friction_torque_dynamic = 0.02

def screw_torque_with__dynamic_friction(w, F):  # [Tmax,T,Tmin]

    pitch_rad = pitch / 2 / np.pi  # mm/rad
    T_no_friction = F * pitch_rad

    R_eff = (screw_outer_dia + screw_inner_dia) / 4  # screw effective radius

    # dynamic friction
    T = T_no_friction + (abs(F) * R_eff * friction_coef_dynamic + no_load_friction_torque_dynamic) * np.sign(w)
    return T


def screw_torque_with_static_friction(F):  # [Tmax,T,Tmin]

    R_eff = (screw_outer_dia + screw_inner_dia) / 4  # screw effective radius

    pitch_rad = pitch / 2 / np.pi  # mm/rad

    T_no_friction = F * pitch_rad

    Tmax = T_no_friction + (abs(F) * R_eff * friction_coef_static + no_load_friction_torque_static)
    Tmin = T_no_friction - (abs(F) * R_eff * friction_coef_static + no_load_friction_torque_static)

    return Tmax, Tmin
