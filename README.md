# Single Neuron Simulation for Generative Purposes Using Izhikevich Model

## Executive Summary

This repository provides a robust framework for simulating and forecasting single neuron dynamics using the Izhikevich model. It supports biophysically accurate simulations of multiple neuron types, systematic parameter sweeps, and advanced generative forecasting using a hybrid of Nixtla TimeGPT and deep learning residual correction. The project is designed for computational neuroscience research and the development of hybrid generative models for neural activity.

---

## Table of Contents

- [Single Neuron Simulation for Generative Purposes Using Izhikevich Model](#single-neuron-simulation-for-generative-purposes-using-izhikevich-model)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [System Architecture](#system-architecture)
  - [Installation](#installation)
  - [Directory Structure](#directory-structure)
  - [Usage](#usage)
    - [Basic Neuron Simulation](#basic-neuron-simulation)
    - [Parameter Variation Study](#parameter-variation-study)
    - [Time Series Forecasting with TimeGPT](#time-series-forecasting-with-timegpt)
    - [Generative Neuron Loop with Residual Forecasting](#generative-neuron-loop-with-residual-forecasting)
  - [Data Output](#data-output)
  - [Visualization](#visualization)
  - [Troubleshooting](#troubleshooting)
  - [References](#references)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)
  - [Author](#author)

---

## Features

- **Biophysically Accurate Simulation:** Implements the Izhikevich neuron model, supporting a wide range of neuron types (regular spiking, fast spiking, bursting, etc.).
- **Parameter Sensitivity Analysis:** Systematic parameter sweeps for all core Izhikevich parameters (`a`, `b`, `c`, `d`, `I`) to visualize their effects on neural dynamics.
- **Time Series Forecasting:** Integrates Nixtla TimeGPT for advanced forecasting of membrane potential trajectories.
- **Residual Correction with Deep Learning:** Employs an LSTM-based residual forecaster to refine TimeGPT predictions in a hybrid generative loop.
- **Automated Data Export:** Simulation and forecast outputs are saved as CSV files for downstream analysis.
- **Comprehensive Visualization:** Generates publication-quality plots comparing simulated, forecasted, and corrected neural activity.

---

## System Architecture

- **Simulation Engine:** Parameterized Izhikevich model in Python, supporting both single-run and batch parameter sweeps.
- **Forecasting Module:**
  - **Primary:** TimeGPT (via Nixtla API) for time series prediction.
  - **Residual:** PyTorch LSTM network trained on forecast residuals for correction.
- **Generative Loop:** Iterative simulation–forecast–residual correction pipeline.
- **Data Management:** Automated directory creation and CSV export for simulations, forecasts, and comparative plots.

---

## Installation

1. **Clone the repository:**

```bash
  git clone https://github.com/aryanbhardwaj24/Single-neuron-simulation-for-generative-purposes-using-Izhikevich-model.git

  cd Single-neuron-simulation-for-generative-purposes-using-Izhikevich-model
```

2. **Install dependencies:**

```
  pip install -r requirements.txt
```

3. **Set up Nixtla API:**

&emsp;&emsp;Register for a [Nixtla (TimeGPT) API key](https://dashboard.nixtla.io/sign_in) and set it in your environment or directly in the code.

---

## Directory Structure

```
.
├── single_neuron.py
├── requirements.txt
├── data/
│ └── .csv # Simulation data for each neuron type
├── forecasts/
│ └── forecast_.csv # TimeGPT forecast outputs
├── comparison_plots/
│ └── *_comparison.png # Plots comparing simulation and forecasts
└── ...
```

- **single_neuron.py**: Main codebase for simulation, forecasting, and generative modeling.
- **requirements.txt**: Python dependencies.
- **data/**: Simulated membrane potential traces (per neuron type).
- **forecasts/**: TimeGPT forecast outputs.
- **comparison_plots/**: Visual comparisons (original vs. forecast vs. corrected).

---

## Usage

### Basic Neuron Simulation

Simulate and visualize canonical neuron types:

```py
python single_neuron.py
```

- Generates and saves membrane potential traces for all supported neuron types.
- Output CSVs are stored in `data/`.

### Parameter Variation Study

Explore the effect of parameter changes on neuron dynamics:

- The script systematically varies each Izhikevich parameter (`a`, `b`, `c`, `d`, `I`) and plots the resulting changes in firing behavior.
- Plots are displayed and can be saved for further analysis.

### Time Series Forecasting with TimeGPT

Forecast future neuronal activity:

- The script prepares simulation data for TimeGPT, sends it via the Nixtla API, and retrieves forecasts with confidence intervals.
- Forecasted data is saved in `forecasts/`.
- Comparative plots (original vs. forecast) are saved in `comparison_plots/`.

### Generative Neuron Loop with Residual Forecasting

Run the advanced generative pipeline:

- Executes an iterative loop:
  - Simulate neuron activity.
  - Forecast future activity using TimeGPT.
  - Compute residuals (simulation - forecast).
  - Train an LSTM-based residual model to correct forecast errors.
  - Update simulation parameters and repeat.
- Plots and data for each cycle are saved for analysis.

---

## Data Output

- **Simulation Data:** CSV files with columns `Time (ms)`, `Membrane Potential (mV)` for each neuron type.
- **Forecast Data:** CSV files with forecasted membrane potentials and confidence intervals.
- **Comparative Plots:** PNG images showing original, forecasted, and corrected traces.

---

## Visualization

- **Neuron Type Plots:** Individual subplots for each neuron type, showing membrane potential over time.
- **Parameter Sweep Plots:** Overlaid traces illustrating the effect of parameter variation.
- **Forecast Comparison Plots:** Side-by-side or overlaid plots of simulation, TimeGPT forecast, LSTM-corrected forecast, and actual future simulation.

---

## Troubleshooting

- **Nixtla API Issues:** Ensure your API key is valid and your network connection is stable.
- **PyTorch Errors:** Verify that PyTorch is installed and CUDA is available if using GPU acceleration.
- **Data Export Problems:** Check directory permissions and existence; the script auto-creates required folders.
- **Forecast Alignment:** Ensure that the simulation and forecast horizons match; the script pads or truncates as needed.

---

## References

- Izhikevich, E. M. (2003). Simple Model of Spiking Neurons. _IEEE Transactions on Neural Networks_, 14(6), 1569–1572.
- Nixtla TimeGPT: https://nixtla.github.io/
- PyTorch: https://pytorch.org/

---

## Acknowledgements

This project was made possible by the continuous guidance and mentorship of **[Dr. Mohan Raghavan](https://iith.ac.in/bme/mohanr/)**, who provided the opportunity and support to pursue this work over a dedicated four-month period. This project also draws on foundational work in computational neuroscience and time series forecasting. Special thanks to the open-source community for tools in scientific computing, machine learning, and neural modeling.

---

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Aryan Bhardwaj**

- [GitHub](https://github.com/aryanbhardwaj24)
- [LinkedIn](https://www.linkedin.com/in/aryanbhardwaj24/)
- [Email](mailto:aryanbhardwaj1328@gmail.com)

---

_For questions, feature requests, or contributions, please open an issue or contact the author directly._
