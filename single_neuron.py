import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from nixtla import NixtlaClient

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

nixtla_client = NixtlaClient(api_key='YOUR_API_KEY', timeout=None)

class ResidualForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(ResidualForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

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

def prepare_data_for_timegpt(time, v, freq='25ms'):
    """
    Convert simulation data to a format suitable for TimeGPT.
    
    Args:
        time: Time array from simulation
        v: Membrane potential array from simulation
        freq: Frequency of the time series
        
    Returns:
        DataFrame formatted for TimeGPT
    """
    # Create DataFrame with proper timestamp format
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=len(time), freq=freq),
        'voltage': v
    })
    
    return df

def forecast_neuron_activity(df, horizon=100, finetune_steps=500):
    """
    Use TimeGPT to forecast future neuronal activity.
    
    Args:
        df: DataFrame with timestamp and voltage columns
        horizon: Number of future time steps to forecast
        finetune_steps: Number of steps to fine-tune the model (0 for zero-shot)
        
    Returns:
        DataFrame with forecasted values
    """
    forecast = nixtla_client.forecast(
        df,
        h=horizon,
        time_col='timestamp',
        target_col='voltage',
        level=[80, 90],
        finetune_steps=finetune_steps,
        freq='25ms'
    )
    return forecast

def plot_forecast(original_df, forecast_df):
    """
    Plot original neuronal activity with forecasted activity.
    
    Args:
        original_df: DataFrame with original data
        forecast_df: DataFrame with forecasted data
    """
    nixtla_client.plot(
        original_df,
        forecast_df,
        time_col='timestamp',
        target_col='voltage'
    )

def neurons_with_timegpt():
    # Define neuron types and their parameter sets
    neuron_types = {
        "Regular Spiking (RS)": (0.02, 0.2, -65, 8, 10),
        "Fast Spiking (FS)": (0.1, 0.2, -65, 2, 10),
        "Intrinsically Bursting (IB)": (0.02, 0.2, -55, 4, 10),
        "Chattering (CH)": (0.02, 0.2, -50, 2, 10)
    }
    
    # Create directories for saving data and forecasts if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('forecasts', exist_ok=True)
    os.makedirs('comparison_plots', exist_ok=True)
    
    # Simulation parameters
    T = 500        # Total simulation time (ms)
    dt = 0.25      # Time step (ms)
    forecast_horizon = 200  # Number of time steps to forecast
    
    for label, params in neuron_types.items():
        a, b, c, d, I = params
        
        # Run simulation
        t, v = izhikevich(a, b, c, d, I, T=T, dt=dt)
        
        # Prepare data for TimeGPT
        df = prepare_data_for_timegpt(t, v)
        
        # Save original simulation data
        filename = label.replace(' ', '_').replace('(', '').replace(')', '') + '.csv'
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        
        # Generate forecast using TimeGPT
        forecast_df = forecast_neuron_activity(df, horizon=forecast_horizon)
        
        # Save forecast data
        forecast_filepath = os.path.join('forecasts', 'forecast_' + filename)
        forecast_df.to_csv(forecast_filepath, index=False)
        
        # Create a new figure for comparison (original vs forecast)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left subplot: Original Simulation
        axs[0].plot(t, v, 'b')
        axs[0].set_title(f"{label} - Original Simulation")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Membrane Potential (mV)")
        axs[0].set_ylim([-80, 35])
        axs[0].grid(True)
        
        # Right subplot: TimeGPT Forecast
        # Extract the last 100 points of original data for context
        context_df = df.iloc[-100:]
        # Plot original context
        axs[1].plot(range(100), context_df['voltage'].values, 'b', label='Original')
        # Plot forecast mean (using the first point as the last observed value)
        axs[1].plot(
            range(99, 100 + forecast_horizon),
            [context_df['voltage'].iloc[-1]] + forecast_df['TimeGPT'].tolist(),
            'r',
            label='Forecast'
        )
        # Plot 80% confidence interval (using TimeGPT-lo-80 and TimeGPT-hi-80)
        axs[1].fill_between(
            range(100, 100 + forecast_horizon),
            forecast_df['TimeGPT-lo-80'].values,
            forecast_df['TimeGPT-hi-80'].values,
            color='r',
            alpha=0.2,
            label='80% Confidence Interval'
        )
        axs[1].set_title(f"{label} - TimeGPT Forecast")
        axs[1].set_xlabel("Time Steps")
        axs[1].set_ylabel("Membrane Potential (mV)")
        axs[1].set_ylim([-80, 35])
        axs[1].legend(fontsize=8)
        axs[1].grid(True)
        
        fig.tight_layout()
        
        # Save the figure to a file
        plot_filename = label.replace(' ', '_').replace('(', '').replace(')', '') + '_comparison.png'
        plot_filepath = os.path.join('comparison_plots', plot_filename)
        fig.savefig(plot_filepath)
        plt.close(fig)

def train_residual_forecaster(model, optimizer, criterion, forecast_series, residual_series, epochs=1000):
    """
    Trains the residual model on the forecast values (features) and corresponding residuals.
    """
    # Convert data to PyTorch tensors and create a dataset/dataloader
    dataset = TensorDataset(torch.FloatTensor(forecast_series), torch.FloatTensor(residual_series))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            # Treat each input as a sequence of length 1 (for simplicity)
            inputs_seq = inputs.unsqueeze(1)  # shape: (batch, seq_len=1, input_size)
            outputs = model(inputs_seq)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Residual Model - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
    return model

def generative_neuron_loop(a, b, c, d, I_initial, cycles=3, T_initial=500, forecast_horizon=200):
    """
    Implements a two-step forecasting system:
    - Primary forecast via TimeGPT.
    - Residual forecasting using an LSTM model.
    The loop simulates neuron activity, forecasts the future, computes the residual error,
    trains a residual model, and corrects the forecast iteratively.
    """
    min_required_points = 144
    T_initial = max(T_initial, min_required_points * 0.25)
    
    # Run the initial simulation
    t, v = izhikevich(a, b, c, d, I_initial, T=T_initial, dt=0.25)
    df = prepare_data_for_timegpt(t, v)
    
    # Lists to store data from each cycle
    all_simulation_dfs = [df]
    all_forecast_dfs = []       # Primary forecasts
    all_corrected_forecasts = []  # Forecasts after residual correction
    
    # Initialize the residual forecasting model
    residual_model = ResidualForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    optimizer = optim.Adam(residual_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Lists to accumulate forecast and residual training data over cycles
    forecast_history = []
    residual_history = []
    
    for cycle in range(cycles):
        print(f"\nCycle {cycle+1} starting...")
        # 1. Generate primary forecast using TimeGPT
        forecast_df = forecast_neuron_activity(df, horizon=forecast_horizon)
        all_forecast_dfs.append(forecast_df)
        
        # Extract the primary forecast (assume the column 'TimeGPT' holds the mean forecast)
        primary_forecast = forecast_df['TimeGPT'].values  # shape (forecast_horizon,)
        
        # 2. Run new simulation with an updated input current
        # Here we use a heuristic: update I based on the mean magnitude of the primary forecast
        new_I = np.abs(primary_forecast).mean() * 0.1 + 5
        new_T = max(forecast_horizon * 0.25, min_required_points * 0.25)
        new_t, new_v = izhikevich(a, b, c, d, new_I, T=new_T, dt=0.25)
        new_df = prepare_data_for_timegpt(new_t, new_v)
        new_df['timestamp'] = pd.date_range(
            start=df['timestamp'].iloc[-1] + pd.Timedelta('25ms'),
            periods=len(new_t),
            freq='25ms'
        )
        
        # 3. Compute the residual error: (Simulated - Primary Forecast)
        # Assume that the first 'forecast_horizon' points of the simulation correspond to the forecast period
        simulated_segment = new_df['voltage'].values[:forecast_horizon]
        if len(simulated_segment) < forecast_horizon:
            simulated_segment = np.pad(simulated_segment, (0, forecast_horizon - len(simulated_segment)), mode='edge')
        residual = simulated_segment - primary_forecast  # shape: (forecast_horizon,)
        
        # 4. Append current cycle's data to the training history
        forecast_history.extend(primary_forecast.reshape(-1, 1))
        residual_history.extend(residual.reshape(-1, 1))
        
        # 5. Train (or update) the residual model if enough data has been collected
        if len(forecast_history) > 10:
            residual_model = train_residual_forecaster(
                residual_model,
                optimizer,
                criterion,
                np.array(forecast_history),
                np.array(residual_history),
                epochs=1000
            )
        
        # 6. Predict the residual correction for the current forecast
        with torch.no_grad():
            forecast_tensor = torch.FloatTensor(primary_forecast.reshape(-1, 1)).unsqueeze(1)  # shape: (forecast_horizon, 1, 1)
            predicted_residual = residual_model(forecast_tensor).squeeze(-1).numpy()
        
        # 7. Correct the forecast by adding the predicted residual
        corrected_forecast = primary_forecast + predicted_residual
        all_corrected_forecasts.append(corrected_forecast)
        
        # Optionally, one might update the simulation input using the corrected forecast instead of the primary one.
        # For now, we continue with the current heuristic.
        
        # 8. Store the new simulation for the next cycle and update df
        all_simulation_dfs.append(new_df)
        df = new_df
    
    # Return the simulation, primary forecast, and corrected forecast data from all cycles
    return all_simulation_dfs, all_forecast_dfs, all_corrected_forecasts

if __name__ == '__main__':
    # Run basic neuron simulations
    neurons_basic()
    
    # Run parameter variation study
    neurons_step()
    
    # Run TimeGPT integration for forecasting
    neurons_with_timegpt()

    # Define Regular Spiking neuron parameters
    a, b, c, d, I = 0.02, 0.2, -65, 8, 10
    
    # Run the advanced generative neuron loop with residual forecasting (2-step forecast)
    sim_dfs, forecast_dfs, corrected_forecasts = generative_neuron_loop(
        a, b, c, d, I, cycles=3, T_initial=500, forecast_horizon=200
    )
    
    # For each cycle, plot context, primary forecast, corrected forecast, and actual simulation.
    for i, (sim_df, forecast_df, corrected_forecast) in enumerate(zip(sim_dfs[1:], forecast_dfs, corrected_forecasts), 1):
       # Use last 100 points from previous simulation as context
        context_df = sim_dfs[i-1].iloc[-100:]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
       # Plot context (blue)
        ax.plot(range(100), context_df['voltage'].values, 'b', label='Context (Simulated)')
        
       # Primary forecast (red)
        forecast_time_steps = range(100, 100 + len(forecast_df) + 1)
        primary_forecast_line = [context_df['voltage'].iloc[-1]] + forecast_df['TimeGPT'].tolist()
        ax.plot(forecast_time_steps, primary_forecast_line, 'r', label='Primary Forecast')
        
       # Corrected forecast (orange)
        corrected_time_steps = range(100, 100 + len(corrected_forecast) + 1)
        corrected_forecast_line = [context_df['voltage'].iloc[-1]] + corrected_forecast.tolist()
        ax.plot(corrected_time_steps, corrected_forecast_line, 'orange', label='Corrected Forecast')
        
       # Actual simulation (green)
        sim_time_steps = range(100, 100 + len(sim_df))
        ax.plot(sim_time_steps, sim_df['voltage'].values, 'g', label='Actual Simulation')
        
        ax.set_title(f'Cycle {i}: Primary vs. Corrected Forecast vs. Actual Simulation')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Membrane Potential (mV)')
        ax.set_ylim([-80, 35])
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        plt.show()
