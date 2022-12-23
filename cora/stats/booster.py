from numpy import ndarray
from typing import Optional
from lightgbm import LGBMRegressor, Dataset, Booster, train as __train__, record_evaluation
from sklearn.model_selection import GridSearchCV, train_test_split
from ..util.config.parser import JSONType


def get_params(override: Optional[JSONType]=None) -> JSONType:
    """

    """
    ans = {
        'objective': 'regression',
        'boost_from_average': False,
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'num_boost_round': 100,
        'subsample': 0.8,
        'feature_fraction': 0.8,
        'force_col_wise': True,
    }
    if override is not None:
        ans.update(**override)
    return ans


def cross_validate(search_params: JSONType, x_data: ndarray, y_data: ndarray, weight: ndarray,
                   cv: int=3, override: Optional[JSONType]=None) -> JSONType:
    """

    """
    # get fixed params
    params = get_params(override=override)
    for key in search_params.keys():
        if key in params:
            del params[key]

    # make estimator and run cv
    estimator = LGBMRegressor(**params)
    search = GridSearchCV(estimator=estimator, param_grid=search_params, cv=cv)
    grid = search.fit(x_data, y_data, sample_weight=weight)

    # now this is the best weight
    params.update(**grid.best_params_)
    return params


def train_with_hold(params: JSONType, x_data: ndarray, y_data: ndarray, weight: ndarray,
                    hold_size: float=.3, plot_metric: bool=False) -> Booster:
    """

    """
    data = train_test_split(x_data, y_data, weight, test_size=hold_size, random_state=0, shuffle=True)
    x_train, x_hold, y_train, y_hold, w_train, w_hold = data
    train_data = Dataset(data=x_train, label=y_train, weight=w_train)
    hold_data = Dataset(data=x_hold, label=y_hold, weight=w_hold, reference=train_data)

    # now evaluate the model
    evals = dict()
    model = __train__(params=params, train_set=train_data, valid_sets=[hold_data, train_data],
                      valid_names=['hold', 'train'], callbacks=[record_evaluation(evals)])
    if plot_metric:
        from lightgbm import plot_metric
        plot_metric(evals)
    return model
