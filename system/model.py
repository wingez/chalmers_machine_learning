from typing import Dict, Callable

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

## Sensor errors ####################
import random
import bisect

rand = np.random

# Sannolikhet för fel per rotation
fault_value = 0.0001

# Set random seed
random.seed(1)

# Lista för snabb Standard Deviation
std_list = [1, 0.318, 0.046, 0.004, 0.002, 0]

# Lista för möjliga utfall av felet
missed_rev_list = [0, 1, 2, 3, 4, 5, 6]

###################################

# Simulation step in seconds
dT = 0.0001

# Trapezoidal screw mechanical properties

screw_outer_dia = 0.024
screw_inner_dia = 0.019
pitch = 0.005  # m/revulution

friction_coef_static = 0.15
friction_coef_dynamic = 0.10

no_load_friction_torque_static = 0.03
no_load_friction_torque_dynamic = 0.02

# System konstants
L = 5 / 1000  # vad ska det vara här? Läste 28 mH nånstans
B = 0.1  # 0 friktion beroende på varvtal
k_lambda = 0.03
R = 0.5

g = 9.82
transmission_reduction = 9

rads_to_ms = pitch / transmission_reduction / 2 / np.pi

J = 0.00024  # moment konstant för motorn + system???

########################
# Konstant för spänningsfiltrering
# (Blir multiplicerad med tidssteget)
filter_step = 20

##################

# Regulalator controller parameters
Kp = 0.3
Ki = 0.05

# Modes the regulator can operate in
REGULATOR = "regulator"
DIRECT_VOLTAGE = "voltage"


# The data which is returned from the simulation which we can use to make graphs
@dataclass
class SimulationResult:
    current: np.ndarray
    omega: np.ndarray
    pos: np.ndarray

    vel: np.ndarray
    acc: np.ndarray

    time: np.ndarray

    ref: np.ndarray

    t_dev: np.ndarray
    t_last: np.ndarray

    voltages: Dict[str, np.ndarray]

    error: np.ndarray
    error_accumulated: np.ndarray

    P_mek: np.ndarray
    P_ut: np.ndarray
    P_e: np.ndarray
    E_broms: np.ndarray
    g_effekt: np.ndarray


def simulate(seconds: float, max_current: float, m: float, voltage_selector: str,
             target_function: Callable[[float], float]) -> SimulationResult:
    # Antal simuleringssteg
    N = int(seconds / dT)

    # Tillståndsvariabler i motorn
    I = np.zeros(N)
    # Varvtal på motorn i rad/s
    omega = np.zeros(N)

    # Rörelse på skrivbordet i m, m/s, m/s2
    # vel är proportionellt mot omega (med en utväxlingskonstant)
    pos = np.zeros(N)
    vel = np.zeros(N)
    acc = np.zeros(N)

    # Genomsnittsspänning för extrauppgift ##

    v_filt = np.zeros(N)

    ######

    # Spänningen till systemet
    V = np.zeros(N)

    # Spänningen över de olika komponenterna i motorn
    vL = np.zeros(N)
    vR = np.zeros(N)
    vEa = np.zeros(N)

    # Vridmoment ut från motorn
    T_dev = np.zeros(N)
    # Vridmoment från last. Omvänd riktining
    T_last = np.zeros(N)

    # Referens-signal
    ref = np.zeros(N)

    # Errors från PI-regulatorn
    Error = np.zeros(N)
    ErrorAccumulated = np.zeros(N)

    # Effekt
    P_mek = np.zeros(N)
    P_ut = np.zeros(N)
    P_e = np.zeros(N)
    E_broms = np.zeros(N)
    g_effekt = np.zeros(N)

    # Loopa varje simuleringssteg
    for i in range(1, N):

        current_time = i * dT

        ref[i] = target_function(current_time)

        # Sensorfel
        # Fel gällande avlästa varv
        if False:  # random.random() <= fault_value * omega[i - 1]:
            rev_fault = omega_sensorfault(omega[i - 1])
        else:
            rev_fault = 0

        # Select how we should calculate the voltage to the motor
        if voltage_selector == DIRECT_VOLTAGE:
            targetVoltage = ref[i]
        elif voltage_selector == REGULATOR:

            # Enkel PI regulator
            target_velocity = ref[i]
            target_omega = target_velocity / rads_to_ms

            last_omega = omega[i - 1] - rev_fault
            Error[i] = target_omega - last_omega
            ErrorAccumulated[i] = ErrorAccumulated[i - 1] + Error[i] * dT
            if target_omega == 0:
                ErrorAccumulated[i] = 0

            targetVoltage = Kp * Error[i] + Ki * ErrorAccumulated[i]
        else:
            raise NotImplementedError()

        # clamp [-24 +24V]
        targetVoltage = max(-24, min(24, targetVoltage))

        # Current sensor: ACS723LLCTR-10AB-T with sensitivity of +/- 1.5%
        # lägga till mätbrus 15mA, mätosäkerhet 1.5%
        last_current = I[i - 1]
        sensor_noise = (0.015 * last_current) + 0.015
        measured_current = last_current + rand.uniform(-sensor_noise, sensor_noise)
        # clamp current So that it dont exceed max-current. Requirement from LArs in mail
        V[i] = targetVoltage if abs(measured_current) < max_current else 0

        # Beräkna spänningarna över de olika delarna i motorn
        vR[i] = R * I[i - 1]
        vEa[i] = k_lambda * omega[i - 1]
        vL[i] = V[i] - vR[i] - vEa[i]

        # Beräkna vridmomentet från motorn
        T_dev[i] = k_lambda * I[i - 1]

        # Beräkna t_last. Dvs motverkande vridmoment från lasten
        applied_force = m * (g + 0)

        # Antag vi står still om omega är nära 0
        is_dynamic = abs(omega[i - 1]) > 1
        if is_dynamic:
            # dynamisk friction
            T_last[i] = screw_torque_with__dynamic_friction(omega[i - 1], applied_force)
            T_last[i] = T_last[i] / transmission_reduction

        else:
            # static friction
            T_last_max, T_last_min = screw_torque_with_static_friction(applied_force)
            T_last_max = T_last_max / transmission_reduction
            T_last_min = T_last_min / transmission_reduction

            if T_last_min <= T_dev[i] <= T_last_max:
                # If T_dev is within the range we should not move at all.
                T_last[i] = T_dev[i]
            else:
                if T_dev[i] > T_last_max:
                    T_last[i] = T_last_max
                else:
                    T_last[i] = T_last_min

        # T_last har hitills beräknats på trapsgängstången.
        # Multiplicera med utväxlingen i växellådan för att få vridmoment på motorn
        # T_last[i] = T_last[i] / transmission_reduction

        # Beräkna strömderivatan
        dI = (V[i] - R * I[i - 1] - k_lambda * omega[i - 1]) / L
        # Integrera strömmen för nästa simuleringssteg
        I[i] = I[i - 1] + dT * dI

        # Beräkna omega-derivatan
        dOmega = (T_dev[i] - T_last[i]) / J
        # Integrera omega för nästa simuleringssteg
        omega[i] = omega[i - 1] + dT * dOmega

        # vel och acc är proportionellt mot omega respektive dOmega
        acc[i] = dOmega * rads_to_ms
        vel[i] = omega[i] * rads_to_ms

        # Positionen är hastigheten integrerat
        pos[i] = pos[i - 1] + dT * vel[i]

        # Beräkna genomsnittsspänning
        v_filt[i] = v_filt[i - 1] * (1 - dT / (filter_step * dT)) + V[i] * dT / (filter_step * dT)

        # Elektrisk effekt
        P_e[i] = V[i] * I[i]
        # Mekanisk effekt
        P_mek[i] = omega[i] * T_dev[i]
        P_ut[i] = applied_force * vel[i]

        E_broms[i] = (m * vel[i] ** 2) / 2 + (1 / 2) * J * omega[i] ** 2

        g_effekt[i] = m * g * pos[i]  # gravitationseffekt, för neråt

    t = np.linspace(0, N * dT, N)
    return SimulationResult(
        current=I,
        omega=omega,
        time=t,
        t_dev=T_dev,
        t_last=T_last,
        pos=pos,
        vel=vel,
        acc=acc,
        ref=ref,
        voltages={
            "V_L": vL,
            "V_R": vR,
            "V_Ea": vEa,
            "v_Supply": V,
            "v_filt": v_filt,
        },
        error=Error,
        error_accumulated=ErrorAccumulated,
        P_mek=P_mek,
        P_ut=P_ut,
        P_e=P_e,
        E_broms=E_broms,
        g_effekt=g_effekt
    )


def screw_torque_with__dynamic_friction(w, F):  # [T]

    pitch_rad = pitch / 2 / np.pi  # (Sträcka per radian)
    T_no_friction = F * pitch_rad  # (N * m per radian)

    # Radien till gängans mittpunkt
    R_eff = (screw_outer_dia + screw_inner_dia) / 4

    # Funktionen nedan ger summan av:
    # Det friktionsfria vridmomentet.
    # Vridmomentet av friktion från gängan under belastning.
    # Det "passiva" vridmomentet som gäller oavsett belastning.
    # Samtliga för det dynamiska fallet och med hänsyn till rotationsriktning.

    T = T_no_friction + (abs(F) * R_eff * friction_coef_dynamic + no_load_friction_torque_dynamic) * np.sign(w)
    return T


def screw_torque_with_static_friction(F):  # [Tmax,T,Tmin]

    R_eff = (screw_outer_dia + screw_inner_dia) / 4  # Gängradie

    pitch_rad = pitch / 2 / np.pi

    T_no_friction = F * pitch_rad

    # Här beräknas båda relevanta 'gränser' för vad vridmomentet
    # behöver överkomma för att gå över till det dynamiska stadiet
    #
    # Beräkningen sker på liknande sätt som ovan men utan rotationsriktning
    # eftersom detta är två olika utfall
    Tmax = T_no_friction + (abs(F) * R_eff * friction_coef_static + no_load_friction_torque_static)
    Tmin = T_no_friction - (abs(F) * R_eff * friction_coef_static + no_load_friction_torque_static)

    return Tmax, Tmin


def omega_sensorfault(w):
    # Fel har inträffat, beräkna hur "allvarligt" felet är enligt standard deviation
    # Denna funktion ger ett index
    fault_index = bisect.bisect_right(std_list, random.random())
    # Av rimlighetsskäl så bör antalet fellästa varv inte bli fler än
    # antalet faktiska varv. Därför görs följande if else.
    if w >= 6:
        rev_fault = missed_rev_list[fault_index] * (random.randint(0, 1) * 2 - 1)
    else:
        if fault_index > int(w):
            fault_index = int(w)

        rev_fault = range(int(w))[fault_index] * (random.randint(0, 1) * 2 - 1)

    return rev_fault
