## ISR spectrum
![ISR spectrum](https://github.com/engeir/code-for-master/workflows/ISR%20spectrum/badge.svg)
### Info
Makes plots of ISR spectrum based on the theory presented in [Hagfors (1961)](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/JZ066i006p01699).

### Installing
Run
```
pip install -r requirements.txt
```
to install all needed packages, and then run with command
```
python3 main.py
```

### File structure
```
/
├── config.py
├── extra/
│   ├── simple_calculations.py
│   └── simple_plots.py
├── integrand_functions.py
├── main.py
├── not_in_use/
│   ├── chirpz.py
│   ├── int_cy.pyx
│   ├── profile.py
│   ├── pure_cython.pyx
│   └── setup.py
├── parallelization.py
├── path.txt
├── paths.py
├── readme.md
├── requirements.txt
├── test_ISR.py
├── tool.py
└── v_int_parallel.py
```
