from typing import List
from numpy import nansum
from pandas import DataFrame, Series
from lightgbm import Booster


class AlphaModel(object):
    """

    """
    @staticmethod
    def load_linear_weight() -> DataFrame:
        """
            for notebook plotting usage
        """
        from glob import glob
        from pandas import read_parquet, concat
        from ..core.state import State
        from ..util.epoch import Date

        State.console_state(filename='../yaml/main.yaml', stdout='ERROR', stderr='ERROR')
        with State.get().config.sub(section='DataFeed/TradeModel') as sub:
            name = sub.as_path(key='filename').format(content='model/linear', event_date='*', extension='parquet.gz')
        ans = list()
        for filename in glob(name):
            data = read_parquet(filename)
            date_string = filename
            for rep in name.split('*'):
                date_string = date_string.replace(rep, '')
            dobj = Date.from_str(dobj=date_string)
            data.reset_index(inplace=True)
            data['event_date'] = dobj.timestamp
            data.set_index(['event_date', 'feature'], inplace=True)
            ans.append(data)
        ans = concat(ans, axis=0, sort=True)
        return ans

    def __init__(self, linear: DataFrame, booster: Booster) -> None:
        """

        """
        self.linear = linear
        self.booster = booster
        self.non_linear_shrinkage = 0.

    @property
    def feature_column(self) -> List[str]: return self.linear.index.to_list()

    def __call__(self, feature: DataFrame, specific: Series) -> DataFrame:
        """

        """
        for col in self.feature_column:
            if col not in feature.columns:
                raise KeyError(f'{col} not found in input data')
        if feature.columns.to_list() != self.feature_column:
            feature = feature.reindex(columns=self.feature_column)
        specific = specific.reindex(index=feature.index, fill_value=specific.max())
        ans = dict()
        for linear_type in ['prior', 'beta', 'coef_']:
            data = feature.values * self.linear[linear_type].values.reshape([1, -1])
            data = Series(data=nansum(data, axis=1), index=feature.index, name=linear_type)
            ans[linear_type] = data
        # non-linear
        x_data = feature.values / specific.values.reshape([-1, 1])
        y_data = self.booster.predict(data=x_data)
        pred = y_data * specific.values
        data = Series(data=pred, index=feature.index, name='nonlinear')
        ans['nonlinear'] = data
        # finally combine
        ans['alpha'] = ans['coef_'] + (1 - self.non_linear_shrinkage) * ans['nonlinear']
        # package
        ans = DataFrame.from_dict(ans, orient='columns')
        assert ans.index.equals(feature.index)
        assert not ans.isnull().sum(axis=0).any()
        return ans
