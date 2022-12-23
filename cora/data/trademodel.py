from glob import glob
from lightgbm import Booster
from pandas import DataFrame, read_parquet
from ..container.alphamodel import AlphaModel
from ..util.epoch import Date
from ..util.file import mkdir, dirname
from .derived import Derived


class TradeModel(Derived):
    """

    """
    @property
    def max_rollback(self) -> int: return self.config.as_int(key='max_rollback', default=300)

    def locate_filename(self, event_date: Date, content: str, extension: str, rollback: int) -> str:
        """

        """
        date_string = event_date.strftime(self.date_format)
        filename = self.filename.format(content=content, event_date='*', extension=extension)
        max_name = self.filename.format(content=content, event_date=date_string, extension=extension)
        filename = [name for name in glob(filename) if name <= max_name]
        if not len(filename):
            raise FileNotFoundError(f'no {content} found prior to {event_date}')
        ans = max(filename)
        # find the implied date
        date_string = ans
        for shape in self.filename.format(content=content, event_date='*', extension=extension).split('*'):
            date_string = date_string.replace(shape, '')
        eval_ = Date.from_str(dobj=date_string)
        tol_ = eval_.roll(freq=self.freq, shift=rollback)
        if tol_ < event_date:
            raise FileNotFoundError(f'latest {content} {ans} is calibrated on {eval_}, too old for {event_date}')
        return ans

    def read_daily(self, event_date: Date) -> AlphaModel:
        """
            can be used for trading as of bod of event_date
        """
        # load linear weights
        filename = self.locate_filename(event_date=event_date, content='model/linear', extension='parquet.gz',
                                        rollback=self.max_rollback)
        linear = read_parquet(filename)
        # load non-linear model
        filename = self.locate_filename(event_date=event_date, content='model/nonlinear', extension='txt',
                                        rollback=self.max_rollback)
        booster = Booster(model_file=filename)
        return AlphaModel(linear=linear, booster=booster)

    def read_with_date(self, dobj: Date, content: str) -> DataFrame: raise RuntimeError('should not hit this line')

    def serialize(self, linear: DataFrame, booster: Booster, event_date: Date) -> None:
        """

        """
        # save linear model
        fmt = dict(content='model/linear', event_date=event_date.strftime(self.date_format), extension='parquet.gz')
        filename = self.filename.format(**fmt)
        mkdir(path=dirname(filename))
        linear.to_parquet(filename, compression='gzip')
        # save spec risk
        fmt = dict(content='model/nonlinear', event_date=event_date.strftime(self.date_format), extension='txt')
        filename = self.filename.format(**fmt)
        mkdir(path=dirname(filename))
        booster.save_model(filename=filename)
        self.logger.info(f'saved trade model available for trading after bod of {event_date}')
