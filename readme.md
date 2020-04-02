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

### Cython
The calculation of the spectrum can also be done through a full numerical solution directly using an arbitrary velocity distribution function. This is implemented in python-code ready to be executed, but also using cython. This needs to be compiled if edited, which is done through the command
```
python3 setup.py build_ext --inplace
```
