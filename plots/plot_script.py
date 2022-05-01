import matplotlib.pyplot as plt
import numpy as np
vals = [0.6608187134502924, 0.6783625730994152, 0.6210526315789474, 0.5789473684210527, 0.44285714285714284]
z = np.arange(-2, 3, 1)

plt.xlabel("z")
plt.ylabel("p")
plt.title("Graph Connectivity vs Latent Factor Tuning")
plt.plot(z, vals[::-1])
plt.savefig("z_p.png")
plt.show()
