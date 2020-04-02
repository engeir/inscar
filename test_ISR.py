import tool


def test_isr_spectrum1():
    a, b = tool.isr_spectrum('maxwell', kappa=6)
    assert isinstance(a, type(b)) and a.shape == b.shape


def test_isr_spectrum2():
    a, b = tool.isr_spectrum('kappa', kappa=4)
    assert isinstance(a, type(b)) and a.shape == b.shape


def test_isr_spectrum3():
    a, b = tool.isr_spectrum('long_calc', kappa=6)
    assert isinstance(a, type(b)) and a.shape == b.shape


def test_H_spectrum():
    a, b = tool.H_spectrum('kappa', test=True, k=7)
    assert isinstance(a, type(b)) and a.shape == b.shape
