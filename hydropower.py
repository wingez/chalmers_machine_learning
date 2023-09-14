import numpy as np
import matplotlib.pyplot as plt


percentages = np.array(range(0, 100 + 1, 5))
flows = np.array([12,10.46,9.75,8.95,8.29,7.93,7.62,7.28,7.03,6.70,6.41,6.11,5.76,5.42,5.22,4.89,4.50,4.08,3.81,3.31,2.18])




randoms = np.random.normal(6.5,2.5,4000)


randoms_y = np.linspace(randoms.min(),randoms.max(),1000)

randoms_percentages = []
for y in randoms_y:
    total = randoms.size
    randoms_percentages.append((randoms>=y).sum()/total)

randoms_percentages=np.array(randoms_percentages)


fig,(ax1,ax2) = plt.subplots(1,2)

n_bins = 30
ax1.hist(randoms, bins=n_bins)


ax2.plot(randoms_percentages*100,randoms_y, label="our distribution")
ax2.plot(percentages,flows, label="real data")


plt.legend()

plt.show()


