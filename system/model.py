from typing import Dict

import numpy as np

from dataclasses import dataclass

dT = 0.01

L = 5000 / 1000  # vad ska det vara här? Läste 28 mH nånstans
B = 0.1  # 0 friktion beroende på varvtal
k_lambda = 0.2
R = 0.5 * 4

g = 9.82
m = 10
utväxlingskonstant = 1800
verkningsgrad = 0.3
J = 1  # moment konstant för motorn + system???


@dataclass
class SimulationResult:
    current: np.ndarray
    omega: np.ndarray

    time: np.ndarray

    voltages: Dict[str,np.ndarray]


def simulate(seconds: int) -> SimulationResult:
    N = int(seconds / dT)

    I = np.zeros(N)
    omega = np.zeros(N)

    V = np.zeros(N)
    V[2000:6000] = 24  # +24 # spännings konstant 24V

    vL = np.zeros(N)
    vR = np.zeros(N)
    vEa = np.zeros(N)

    for i in range(1, N):
        vR[i] = R * I[i - 1]
        vEa[i] = k_lambda *omega[i - 1]
        vL[i] = V[i] - vR[i] - vEa[i]

        dI = (V[i - 1] - R * I[i - 1] - k_lambda * omega[i - 1]) / L

        I[i] = I[i - 1] + dT * dI

        dOmega = (k_lambda * I[i - 1] - (B * omega[i - 1] + m * g / utväxlingskonstant / verkningsgrad)) / J
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
