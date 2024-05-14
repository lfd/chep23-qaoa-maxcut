# CHEP 2023

QAOA for MaxCut using noise models with varying gate error rate.

Code accompanying the paper
```
@article{franz:23:chep,
  author = {Maja Franz and Pìa Zurita and Markus Diefenthaler and Wolfgang Mauerer},
  title = {Co-Design of Quantum Hardware and Algorithms in Nuclear and High Energy Physics},
  year = {2024}
  doi = {10.1051/epjconf/202429512002},
  journal = {EPJ Web of Conf.},
  pages = {12002},
  url = {https://doi.org/10.1051/epjconf/202429512002},
  userd = {CHEP '23},
  volume = {295},
}
```
## Setup
The code was tested on the [Qaptiva 800s](https://atos.net/en/solutions/high-performance-computing-hpc/quantum-computing-qaptiva) (V1.9.1) with Python 3.9.
The Qaptiva system is required to execute the code.
Additional libraries can be installed via pip. Creating a virtual environment is recommended:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
