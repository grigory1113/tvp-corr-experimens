# Replication Code for: Bayesian Algorithm for Dynamic Correlation Estimation in Economic Time Series
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18891231.svg)](https://doi.org/10.5281/zenodo.18891231)

This repository contains the code to reproduce all experiments and figures from the paper:

> **Bayesian Algorithm for Dynamic Correlation Estimation in Economic Time Series**  
> Nikita Moiseev¹, Grigory Aivazian¹, Alexey Mikhaylov², Nora Baranyai³,*  
> ¹Department of Mathematical Methods in Economics, Plekhanov Russian University of Economics, Moscow, Russia  
> ²Financial University under the Government of the Russian Federation, Moscow, Russia  
> ³Faculty of Engineering, University of Pannonia, Veszprém, Hungary  
> *Corresponding author

The experiments compare the proposed Bayesian Time‑Varying Parameter (TVP) model with DCC‑GARCH and rolling window benchmarks on two real‑world datasets:
- **Oil & Ruble**: dynamic correlation between Brent oil prices and the USD/RUB exchange rate.
- **OFZ & Ruble**: dynamic correlation between Russian government bond (OFZ) yields and the USD/RUB exchange rate.

The core TVP algorithm is implemented in a separate Python package: [`tvp-correlation`](https://github.com/grigory1113/tvp-correlation).

---

**Note:** The folder `src/real_data_experiments` contains the main experiment code. The `experiment/` folder is a legacy directory from earlier versions; it is kept to avoid breaking imports and will be cleaned in future updates.

---

## Requirements

- Python 3.9 or higher
- Core TVP algorithm package: [`tvp-correlation`](https://github.com/grigory1113/tvp-correlation) (install separately)
- Additional Python packages listed in `requirements.txt`

---

## Reproducing the Experiments

### Oil & Ruble experiment

```bash
python -m src.real_data_experiments.oil_ruble_experiment
```

### OFZ & Ruble experiment

```bash
python -m src.real_data_experiments.ofz_ruble_experiment
```

### Synthetic experiment

```bash
python -m src.synthetic_data_experiments.main
```

