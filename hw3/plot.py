import matplotlib.pyplot as plt

with open('value_iteration.txt') as f:
    lines = f.readlines()
    distances = [float(line.strip()) for line in lines]
    iterations = range(len(distances))
    
fig, ax = plt.subplots()
ax.plot(iterations, distances)

ax.set(xlabel='iterations (i)', ylabel='distance',
       title='Value Iteration Convergence')
ax.grid()

fig.savefig("value_iteration_plot.png")
plt.show()