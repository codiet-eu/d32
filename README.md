# Causal Model Learning with Non-Commutative Polynomial Optimization

Welcome to the repository for causal discovery deliverable 3.2! This codebase encompasses notebooks and test results associated with each figure in the deliverable.

## Causal Learning Process
1. **Generate Data:** Use the `IIDSimulation` function in `Generate Data.py` to create artificial true causal graphs and observation data.
2. **Learn Structure:** Uncover the causal structure beneath the observation data.
3. **Visualize Comparison:** Generate heat maps to compare estimated and true graphs.
4. **Calculate Metrics:** Assess the performance metrics.
5. **Demonstrate in Heatmap:** Illustrate results in a heatmap for multiple datasets.

## Getting Started
1. **Synthetic Data:** Utilize the `Ancpop_Synthetic` class to generate custom datasets.
2. **Real-world Data:** Employ the `Ancpop_Real` class in `Ancpop_Real.py` to test real data from the `./Real data` folder. Alternatively, use generated data in `./Synthetic data`.
3. **Experiments:** Experiment scripts for proposed methods and baselines are available in Jupyter notebooks named "Methods_withdevice.ipynb".
4. **Results:** Access all results and figures mentioned in the paper from the `./result` folder. Repeat the results using "ANCPOP_test.ipynb".

## Running Notebooks
- **ANCPOP_test.ipynb:** Execute this notebook for all experiments on synthetic and real-world data using the ANM_NCPOP approach. Upload datasets and the notebook to [Colab](https://colab.research.google.com/) for seamless execution.

## Installation
To run experiments locally, ensure you have the following dependencies installed:
- Python (>= 3.6, <= 3.9)
- tqdm (>= 4.48.2)
- NumPy (>= 1.19.1)
- Pandas (>= 0.22.0)
- SciPy (>= 1.7.3)
- scikit-learn (>= 0.21.1)
- Matplotlib (>= 2.1.2)
- NetworkX (>= 2.5)
- PyTorch (>= 1.9.0)
- ncpol2sdpa 1.12.2 [Documentation](https://ncpol2sdpa.readthedocs.io/en/stable/index.html)
- MOSEK (>= 9.3) [MOSEK](https://www.mosek.com/)
- gcastle (>= 1.0.3)

### PIP Installation
```bash
# Execute the following commands to run the notebook directly in Colab. Ensure your MOSEK license file is in one of these locations:
#
# /content/mosek.lic   or   /root/mosek/mosek.lic
#
# inside this notebook's internal filesystem.
# Install MOSEK and ncpol2sdpa if not already installed
pip install mosek 
pip install ncpol2sdpa
pip install gcastle==1.0.3
```
