import tool


def test_isr_spectrum():
    a, b = tool.isr_spectrum('kappa', kappa=6)
    assert isinstance(a, type(b)) and a.shape == b.shape


def test_H_spectrum():
    a, b = tool.H_spectrum('kappa', test=True, k=7)
    assert isinstance(a, type(b)) and a.shape == b.shape
