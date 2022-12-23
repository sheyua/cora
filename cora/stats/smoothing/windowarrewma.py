from numpy import ndarray, power, append, zeros, isfinite, logical_not, maximum, nansum
from ...util.compare import le, gt
from ...util.constant import NAN


class WindowArrEWMA(object):
    """
        window moving average array version
        infinite values can
            raise   value error
            zero    treat as zero
            count   do not weight it in the mean
    """
    def __init__(self, alpha: float, size: int, window: int, infinite: str='raise') -> None:
        """

        """
        assert window > 0
        assert infinite
        assert gt(alpha, 0.) and le(alpha, 1.)
        assert infinite in ['raise', 'zero', 'count']

        self.size = size
        self.window = window
        self.alpha = alpha
        self.infinite = infinite
        # in reverse order
        self.decay_rate = power(1. - self.alpha, range(self.window - 1, -1, -1))
        # cache shuffled weights
        self._weight = dict()
        for idx in range(self.window):
            self._weight[idx] = append(self.decay_rate[(self.window - idx):], self.decay_rate[:(self.window - idx)])
        # state variables
        self._num = 0
        self._idx = 0
        self._buffer = zeros((self.window, self.size), dtype=float)

    def __call__(self, value: ndarray) -> ndarray:
        """

        """
        assert value.shape == (self.size, )

        # nan treatment
        notnan = isfinite(value)
        isnull = logical_not(notnan)
        if isnull.sum():
            if self.infinite == 'raise':
                raise ValueError('non-finite value encounter')
            elif self.infinite == 'zero':
                value = value.copy()
                value[isnull] = 0.
            elif self.infinite == 'count':
                value = value.copy()
                # make sure all inf became nan
                value[isnull] = NAN
            else:
                raise NotImplementedError
        self._num += 1

        # update buffer
        self._buffer[self._idx, :] = value
        self._idx = (self._idx + 1) % self.window
        return self._value()

    def _value(self) -> ndarray:
        """

        """
        # handle edge-case
        if self._num == 0:
            return zeros((self.size, ), dtype=float)

        # grabbing the weights and values
        if self._num < self.window:
            weight = self.decay_rate[(self.window - self._idx):]
            value = self._buffer[:self._idx, :]
        else:
            value = self._buffer
            weight = self._weight[self._idx]

        # nan treatment
        notnan = isfinite(value)
        # propagate weight with nans escaped
        weight = notnan * weight.reshape([-1, 1])
        # this will help avoid zero division error
        weight_sum = maximum(weight.sum(axis=0), power(1. - self.alpha, self.window))
        # this will skips the nans
        ans = nansum(weight * value, axis=0) / weight_sum
        # now patch nan for count type
        patch = notnan.sum(axis=0) == 0
        if patch.any():
            assert self.infinite == 'count'
            ans[patch] = NAN
        return ans

    def reset(self) -> None:
        """

        """
        self._num = 0
        self._idx = 0
        self._buffer = zeros((self.window, self.size), dtype=float)

    @property
    def value(self) -> ndarray: return self._value()

    @property
    def variance(self) -> ndarray: raise NotImplementedError

    @property
    def num_sample(self) -> int: return self._num
