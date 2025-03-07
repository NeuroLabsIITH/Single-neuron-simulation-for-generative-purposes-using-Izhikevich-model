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

def neurons_basic():  
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
      t, v = izhikevich_basic(a, b, c, d, I, T=200, dt=0.25)
      
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

def neurons_step():
    # Baseline parameters for a Regular Spiking (RS) neuron
    a0, b0, c0, d0, I0 = 0.02, 0.2, -65, 8, 10

    # Define offsets: we will add an increment from 0 to 0.09 (10 steps) for each parameter
    offsets = np.linspace(0, 0.09, 10)

    # Prepare a figure with 5 subplots, one for each parameter variation
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    # Use a colormap to choose distinct colors
    cmap = plt.get_cmap('viridis', len(offsets))

    # 1. Varying a (time scale of u)
    for i, off in enumerate(offsets):
        a = a0 + off
        time, v_vals = izhikevich(a, b0, c0, d0, I0, T=100, dt=1)
        axs[0].plot(time, v_vals, color=cmap(i), label=f"a={a:.3f}")
    axs[0].set_title("Effect of Varying 'a' (Recovery time scale)")
    axs[0].set_ylabel("Membrane Potential (mV)")
    axs[0].grid(True)
    axs[0].legend(loc='upper right', fontsize=8)

    # 2. Varying b (sensitivity of u to v)
    for i, off in enumerate(offsets):
        b = b0 + off
        time, v_vals = izhikevich(a0, b, c0, d0, I0, T=100, dt=1)
        axs[1].plot(time, v_vals, color=cmap(i), label=f"b={b:.3f}")
    axs[1].set_title("Effect of Varying 'b' (Sensitivity to subthreshold fluctuations)")
    axs[1].set_ylabel("Membrane Potential (mV)")
    axs[1].grid(True)
    axs[1].legend(loc='upper right', fontsize=8)

    # 3. Varying c (voltage reset after spike)
    for i, off in enumerate(offsets):
        c = c0 + off  # note: since c is negative, adding a positive offset makes it less negative (shallower reset)
        time, v_vals = izhikevich(a0, b0, c, d0, I0, T=100, dt=1)
        axs[2].plot(time, v_vals, color=cmap(i), label=f"c={c:.1f}")
    axs[2].set_title("Effect of Varying 'c' (After-spike reset voltage)")
    axs[2].set_ylabel("Membrane Potential (mV)")
    axs[2].grid(True)
    axs[2].legend(loc='upper right', fontsize=8)

    # 4. Varying d (after-spike jump in recovery variable)
    for i, off in enumerate(offsets):
        d = d0 + off
        time, v_vals = izhikevich(a0, b0, c0, d, I0, T=100, dt=1)
        axs[3].plot(time, v_vals, color=cmap(i), label=f"d={d:.2f}")
    axs[3].set_title("Effect of Varying 'd' (After-spike jump of u)")
    axs[3].set_ylabel("Membrane Potential (mV)")
    axs[3].grid(True)
    axs[3].legend(loc='upper right', fontsize=8)

    # 5. Varying I (External input current)
    for i, off in enumerate(offsets):
        I = I0 + off
        time, v_vals = izhikevich(a0, b0, c0, d0, I, T=100, dt=1)
        axs[4].plot(time, v_vals, color=cmap(i), label=f"I={I:.2f}")
    axs[4].set_title("Effect of Varying 'I' (External current)")
    axs[4].set_ylabel("Membrane Potential (mV)")
    axs[4].set_xlabel("Time (ms)")
    axs[4].grid(True)
    axs[4].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # neurons_basic()
    neurons_step()
