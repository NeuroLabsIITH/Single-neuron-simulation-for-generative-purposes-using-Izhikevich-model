import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def izhikevich(a, b, c, d, I, T=1000, dt=0.25):
    """
    Simulate the Izhikevich neuron model.
    Returns the time vector and membrane potential vector.
    """
    n = int(T/dt)
    time = np.linspace(0, T, n)
    v = np.empty(n)
    u = np.empty(n)
    # Initial conditions
    v[0] = -65
    u[0] = b * v[0]
    
    for i in range(n-1):
        dv = (0.04*v[i]**2 + 5*v[i] + 140 - u[i] + I) * dt
        du = a * (b*v[i] - u[i]) * dt
        v[i+1] = v[i] + dv
        u[i+1] = u[i] + du
        if v[i+1] >= 30:
            v[i] = 30  # Set spike peak for plotting
            v[i+1] = c
            u[i+1] = u[i+1] + d
    return time, v

# Define neuron types and their parameter sets:
neuron_types = {
    "Regular Spiking (RS)": (0.02, 0.2, -65, 8, 10),
    "Fast Spiking (FS)": (0.1, 0.2, -65, 2, 10),
    "Intrinsically Bursting (IB)": (0.02, 0.2, -55, 4, 10),
    "Chattering (CH)": (0.02, 0.2, -50, 2, 10),
    "Low-Threshold Spiking (LTS)": (0.02, 0.25, -65, 2, 10),
    "Rebound Spiking": (0.03, 0.25, -60, 4, 10),
    "Phasic Spiking": (0.02, 0.25, -65, 6, 10),
    "Spike Latency": (0.02, 0.2, -65, 8, 8),  # Lower I to induce latency
    "Threshold Variability": (0.03, 0.25, -60, 4, 10),
    "Bistable": (0.02, 0.2, -55, 10, 10),
    "Rebound Burst": (0.03, 0.25, -55, 4, 10),
    "Mixed Mode": (0.02, 0.2, -55, 4, 10)
}

# Create a directory for saving data
os.makedirs('data', exist_ok=True)

plt.figure(figsize=(12, 16))
for i, (label, params) in enumerate(neuron_types.items(), 1):
    a, b, c, d, I = params
    # Run simulation for T=200 ms for clarity in plots
    t, v = izhikevich(a, b, c, d, I, T=200, dt=0.25)
    
    # Plotting
    plt.subplot(6, 2, i)
    plt.plot(t, v, 'b')
    plt.title(label)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.ylim([-80, 35])
    plt.grid(True)
    
    # Save simulation data to CSV file
    df = pd.DataFrame({'Time (ms)': t, 'Membrane Potential (mV)': v})
    # Create a filename from the label (remove spaces and parentheses)
    filename = label.replace(' ', '_').replace('(', '').replace(')', '') + '.csv'
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)

plt.tight_layout()
plt.show()
