from numpy import power


def compute_alpha(halflife: int) -> float:
    """
        alpha is the learning rate
    """
    assert halflife > 0
    return 1 - power(0.5, 1 / halflife)
