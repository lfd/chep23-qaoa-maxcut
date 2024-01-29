# CHEP 2023

QAOA for MaxCut using noise models with varying gate error rate.

Code accompanying the paper
```
@inproceedings{franz23:chep:preprint,
  author = {Maja Franz and Pìa Zurita and Markus Diefenthaler and Wolfgang Mauerer},
  title = {Co-Design of Quantum Hardware and Algorithms in Nuclear and High Energy Physics},
  booktitle = {Proceedings of the 26th International Conference on Computing in High Energy \& Nuclear Physics},
  year = {2023},
  note = {to appear},
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
