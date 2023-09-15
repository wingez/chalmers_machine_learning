import numpy as np
import matplotlib.pyplot as plt

percentages = np.array(range(0, 100 + 1, 5))
flows_river_a = np.array(
    [12, 10.46, 9.75, 8.95, 8.29, 7.93, 7.62, 7.28, 7.03, 6.70, 6.41, 6.11, 5.76, 5.42, 5.22, 4.89, 4.50, 4.08, 3.81,
     3.31, 2.18])
flows_river_b = np.array(
    [19.2, 14.15, 11.56, 10.01, 9.18, 8.63, 8.11, 7.59, 6.9, 6.21, 5.69, 5.45, 5.18, 4.73, 4.31, 3.45, 2.76, 2.42, 2.14,
     1.79, 1.56])

river_a_distrib = np.random.normal(6.5, 2.5, 4000)
river_b_distrib = np.random.normal(4.5, 4.5, 4000)

river_a_distrib = river_a_distrib[river_a_distrib >= 2.5]
river_b_distrib = river_b_distrib[river_b_distrib >= 1.5]

both_rivers = np.concatenate([river_a_distrib, river_b_distrib])
flows_y_axis = np.linspace(both_rivers.min(), both_rivers.max(), 1000)


def sum_flows(dist, flows):
    randoms_percentages = []
    for y in flows:
        total = dist.size
        randoms_percentages.append((dist >= y).sum() / total)

    return np.array(randoms_percentages)


river_a_percentages = sum_flows(river_a_distrib, flows_y_axis)
river_b_percentages = sum_flows(river_b_distrib, flows_y_axis)

fig, (ax1, ax2) = plt.subplots(1, 2)

n_bins = 30
ax1.hist(river_a_distrib, bins=n_bins, alpha=0.5, label="A")
ax1.hist(river_b_distrib, bins=n_bins, alpha=0.5, label="B")

ax1.set(xlabel='flow  (m3/s)', ylabel='frequency',
        title='River flow distribution')

ax1.legend()

ax2.plot(river_a_percentages * 100, flows_y_axis, label="A distrib")
ax2.plot(percentages, flows_river_a, label="A real")

ax2.plot(river_b_percentages * 100, flows_y_axis, label="B distrib")
ax2.plot(percentages, flows_river_b, label="B real")

ax2.set(xlabel='percentage time of flow equal or exceed', ylabel='flow (m3/s)',
        title='Flow duration curves')

ax2.legend()

river_a_avg_flow = river_a_distrib.sum() / river_a_distrib.size
river_b_avg_flow = river_b_distrib.sum() / river_b_distrib.size

print(f"River A total yearly flow: {river_a_avg_flow * (3600 * 24 * 365):.2f} m3", )
print(f"River A avg flow: {river_a_avg_flow:.2f} m3/s")
print(f"River B total yearly flow: {river_b_avg_flow * (3600 * 24 * 365):.2f} m3", )
print(f"River B avg flow: {river_b_avg_flow:.2f} m3/s")

plt.show()

turbine_efficiency_francis = np.array(
    [0.00, 0.01, 0.20, 0.36, 0.49, 0.60, 0.68, 0.75, 0.80, 0.83, 0.86, 0.87, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89, 0.88,
     0.87, 0.86])
turbine_efficiency_pelton = np.array(
    [0.00, 0.14, 0.46, 0.64, 0.75, 0.81, 0.85, 0.86, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87,
     0.87, 0.86])
turbine_efficiency_prop = np.array(
    [0.00, 0.00, 0.00, 0.00, 0.03, 0.09, 0.15, 0.21, 0.27, 0.33, 0.39, 0.45, 0.50, 0.56, 0.62, 0.67, 0.72, 0.77, 0.82,
     0.87, 0.91])


def turb_efficiency(flow, turbine):
    turbine_percentages = np.linspace(0, 100, turbine_efficiency_prop.size)
    return np.interp(flow, turbine_percentages, turbine, )


turb_flows = np.linspace(0, 100, 10000)

fig, ax = plt.subplots()
ax.plot(turb_flows, turb_efficiency(turb_flows, turbine_efficiency_francis), label="Francis")
ax.plot(turb_flows, turb_efficiency(turb_flows, turbine_efficiency_pelton), label="Pelton")
ax.plot(turb_flows, turb_efficiency(turb_flows, turbine_efficiency_prop), label="Prop")

ax.set(ylabel='efficiency (%)', xlabel='% of design flow',
       title='Propeller efficiency')


@np.vectorize
def dual_turb_effiency_francis(flow):
    turb = turbine_efficiency_francis
    if flow <= 1 / 3:
        # only use small turbine
        return turb_efficiency(flow * 3, turb)

    else:
        # use both
        efficiency_small = turb_efficiency(1, turb)
        percentage_small = (1 / 3) / flow

        flow_large = flow - 1 / 3
        flow_large /= (2 / 3)

        efficiency_large = turb_efficiency(flow_large, turb)
        percentage_large = 1 - percentage_small

        return efficiency_small * percentage_small + efficiency_large * percentage_large


ax.plot(turb_flows, dual_turb_effiency_francis(turb_flows), label="dual Francis")

ax.legend()

plt.show()

# generated power

designed_flow = 7.6
Hg = 110
e_t = 0.96
e_g = 0.96
p = 1000
g = 9.82


def power(flow):
    flow_clipped = np.clip(flow, 0, designed_flow)
    return p * g * np.clip(flow, 0, designed_flow) * Hg * e_g * turb_efficiency(flow_clipped / designed_flow * 100,
                                                                                turbine_efficiency_francis)


max_power = power(designed_flow)
print(f"max power in MW: {max_power / 1e6:.2f}", )

river_a_power = power(river_a_distrib)
river_a_power_distrib = np.linspace(0, river_a_power.max(), 1000)
river_a_power_percentages = sum_flows(river_a_power, river_a_power_distrib)

fig, ax = plt.subplots()
ax.plot(river_a_power_percentages * 100, river_a_power_distrib / 1e6, label="A distrib")

ax.set(ylabel='Power (MW)', xlabel='% of time equal or exceeds',
       title='Power duration distribution')

ax.legend()

plt.show()

power_average = np.average(river_a_power) * e_t
energy_per_year_WH = power_average * 24 * 365
print(f"Yearly power production: {energy_per_year_WH / 1e6:.2f} MWh")

CF = energy_per_year_WH / (24 * 365 * max_power)
print(f"capacity factor: {CF:.2f}")

temp = np.sort(river_a_distrib)
temp_flipped = np.flip(temp.copy())
dam_inflow = np.concatenate([temp, temp_flipped, temp, temp_flipped])
dam_content = np.zeros(dam_inflow.size)
dam_outflow = np.zeros(dam_inflow.size)

for i in range(1, dam_inflow.size):
    fill = dam_content[i - 1] + dam_inflow[i]
    if fill >= designed_flow:
        out_flow = designed_flow
        fill -= out_flow
    else:
        out_flow = fill
        fill = 0

    dam_content[i] = fill
    dam_outflow[i] = out_flow

# Create some mock data
date_values = np.linspace(0, dam_inflow.size, dam_inflow.size)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('flow (m3/s)', color=color)
ax1.plot(date_values, dam_inflow, label="inflow")
ax1.plot(date_values, dam_outflow, label="outflow")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(title=f"River Dammed with maximum output flow = {designed_flow} m3/s")

ax1.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax2.set_ylabel('dam fill (m3)', color=color)  # we already handled the x-label with ax1
ax2.plot(date_values, dam_content, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()




river_a_power_dammed = power(dam_outflow)
river_a_power_dammed_distrib = np.linspace(0, river_a_power_dammed.max(), 1000)
river_a_power_dammed_percentages = sum_flows(river_a_power_dammed, river_a_power_dammed_distrib)

fig, ax = plt.subplots()
ax.plot(river_a_power_dammed_percentages * 100, river_a_power_dammed_distrib / 1e6, label="With DAM")
ax.plot(river_a_power_percentages * 100, river_a_power_distrib / 1e6, label="Without DAM")

ax.set(ylabel='Power (MW)', xlabel='% of time equal or exceeds',
       title='Power duration distribution DAMM')

ax.legend()

plt.show()

power_average_dammed = np.average(river_a_power_dammed) * e_t
energy_per_year_WH_dammed = power_average_dammed * 24 * 365
print(f"Yearly power production DAM: {energy_per_year_WH_dammed / 1e6:.2f} MWh")

CF = energy_per_year_WH_dammed / (24 * 365 * max_power)
print(f"capacity factor DAM: {CF:.2f}")






