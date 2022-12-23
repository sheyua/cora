from typing import Optional
from numpy import ascontiguousarray, allclose, maximum    # , identity
from numpy import ndarray, power, array, matmul, sqrt, zeros, diag
from numpy.linalg import cholesky, eigh, LinAlgError
from scipy.linalg import lapack, eigh
from ..util.constant import EPSILON


def invert(arr: ndarray) -> ndarray:
    """

    """
    if not len(arr):
        return arr.copy()
    else:
        lower_triag = cholesky(arr)     # lower triag is c-contiguous

    # lower_triag = asfortranarray(lower_triag)
    lower_inv, status = lapack.dtrtri(lower_triag, lower=1, unitdiag=0, overwrite_c=0)  # lower_inv is f-contiguous
    if status:
        raise LinAlgError('error inverting lower triangular matrix')
    # # solve_triangular is particularly slow
    # lower_inv = solve_triangular(lower_triag, identity(len(arr)), lower=True)

    ans = matmul(lower_inv.T, lower_inv)    # ans should be back to c-contiguous
    if not ans.data.c_contiguous:
        import warnings
        warnings.warn(message='inverting positive definite matrix yields in fortran-contiguous result',
                      category=UserWarning)
        ans = ascontiguousarray(ans)
    return ans


def max_entropy(data: ndarray, prior: Optional[ndarray]=None) -> ndarray:
    """
        prior is positive definite, data is semi-positive definite
        https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/old_IDAPILecture16.pdf
    """
    # sanity check
    if not allclose(data, data.T, rtol=0, atol=EPSILON):
        raise ValueError('input matrix is not symmetric')
    if prior is None:
        prior = diag(diag(data)) * 1 / 3 + data * 2 / 3
    elif not allclose(prior, prior.T, rtol=0, atol=EPSILON):
        raise ValueError('input prior is not symmetric')

    # flat blend
    flat = data + prior
    # find transformation phi
    eig, phi = eigh(flat)

    # obtain diagonal terms
    diag_data = matmul(data, phi)
    diag_data = diag(matmul(phi.T, diag_data))
    diag_prior = matmul(prior, phi)
    diag_prior = diag(matmul(phi.T, diag_prior))

    # maximize entropy
    eig_mix = maximum(diag_data, diag_prior)
    # find the mix
    ans = matmul(diag(eig_mix), phi.T)
    ans = matmul(phi, ans)
    return ans


def newey_west(data: ndarray, weight: ndarray, lag: Optional[int]=None) -> ndarray:
    """
        aggregate whole period volatility considering auto-correlation
    """
    num_obs, num_asset = data.shape
    if lag is None:
        lag = int(4 * power(num_obs / 100, 2 / 9))  # Tsay's time-series book
    assert weight.shape == (num_obs, )

    # first estimat the weighted mean
    mean = data * weight.reshape([-1, 1])
    mean = mean.sum(axis=0)
    data = data - mean.reshape([1, -1])

    # now make sqrt of the weight to simplify
    data = data * sqrt(weight).reshape([-1, 1])

    # now use sqrt weight to transform the
    ans = zeros([num_asset, num_asset])
    for idx in range(lag + 1):
        if idx == 0:
            value = matmul(data.T, data)
        else:
            x_data = data[idx:]
            y_data = data[:-idx]
            value = matmul(y_data.T, x_data) + matmul(x_data.T, y_data)
            # lag adjustment
            value *= 1 - idx / (lag + 1)
        ans += value
    return ans
