import tool


def test_isr_spectrum():
    a, b = tool.isr_spectrum('hagfors')
    assert isinstance(a, type(b)) and a.shape == b.shape


def test_H_spectrum():
    a, b = tool.H_spectrum('kappa', test=True)
    assert isinstance(a, type(b)) and a.shape == b.shape
