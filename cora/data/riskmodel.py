from glob import glob
from pandas import DataFrame, Series, read_parquet, merge
from ..container.factormodel import FactorModel
from ..util.epoch import Date
from ..util.file import mkdir, dirname
from .assetdata import AssetData
from .marketdata import MarketData
from .derived import Derived


class RiskModel(Derived):
    """

    """
    @property
    def max_rollback(self) -> int: return self.config.as_int(key='max_rollback', default=30)

    @property
    def record_start(self) -> Date:
        """

        """
        filename = self.filename.format(content='risk/specific', event_date='*')
        # find the implied date
        date_string = min(glob(filename))
        for shape in self.filename.format(content='risk/specific', event_date='*').split('*'):
            date_string = date_string.replace(shape, '')
        ans = Date.from_str(dobj=date_string)
        return ans

    def locate_filename(self, event_date: Date, content: str, rollback: int) -> str:
        """

        """
        filename = self.filename.format(content=content, event_date='*')
        max_name = self.filename.format(content=content, event_date=event_date.strftime(self.date_format))
        filename = [name for name in glob(filename) if name <= max_name]
        if not len(filename):
            raise FileNotFoundError(f'no {content} found prior to {event_date}')
        ans = max(filename)
        # find the implied date
        date_string = ans
        for shape in self.filename.format(content=content, event_date='*').split('*'):
            date_string = date_string.replace(shape, '')
        eval_ = Date.from_str(dobj=date_string)
        tol_ = eval_.roll(freq=self.freq, shift=rollback)
        if tol_ < event_date:
            raise FileNotFoundError(f'latest {content} {ans} is calibrated on {eval_}, too old for {event_date}')
        return ans

    def read_exposure(self, event_date: Date) -> DataFrame:
        """

        """
        try:
            filename = self.locate_filename(event_date=event_date, content='risk/exposure', rollback=0)
        except FileNotFoundError:
            # get factor exposure this is data_date convention
            data_date = event_date.roll(freq=self.freq, shift=-1)
            feed = AssetData()
            style = feed.read_style_exposure(start_date=data_date, end_date=data_date).loc[data_date.timestamp]
            feed = MarketData()
            sector = feed.read_sector_exposure(start_date=data_date, end_date=data_date).loc[data_date.timestamp]
            ans = merge(style, sector, left_index=True, right_index=True, how='inner')
            ans.sort_index(axis=0, inplace=True)
            # save this asset exposure for quick evaluation
            filename = self.filename.format(content='risk/exposure', event_date=event_date.strftime(self.date_format))
            mkdir(path=dirname(filename))
            ans.to_parquet(filename, compression='gzip')
            self.logger.info(f'cached factor exposure for trading bod of {event_date}')
            return ans
        else:
            ans = read_parquet(filename)
            return ans

    def read_specific(self, event_date: Date) -> Series:
        """

        """
        filename = self.locate_filename(event_date=event_date, content='risk/specific', rollback=self.max_rollback)
        ans = read_parquet(filename).squeeze()
        return ans

    def read_factor_covar(self, event_date: Date) -> DataFrame:
        """

        """
        filename = self.locate_filename(event_date=event_date, content='risk/factor_covar', rollback=self.max_rollback)
        ans = read_parquet(filename)
        return ans

    def read_daily(self, event_date: Date) -> FactorModel:
        """
            can be used for trading as of bod of event_date
        """
        # get factor covariance, exposure, and specific data
        factor_covar = self.read_factor_covar(event_date=event_date)
        specific = self.read_specific(event_date=event_date)
        exposure = self.read_exposure(event_date=event_date)

        # trim index
        index = exposure.index.intersection(specific.index).sort_values()
        columns = factor_covar.columns.difference(exposure.columns)
        if len(columns) != 0:
            raise ValueError(f'exposure missing factor {columns.tolist()}')
        specific = specific.reindex(index=index)
        exposure = exposure.reindex(index=index, columns=factor_covar.columns)
        return FactorModel(exposure=exposure, factor_covar=factor_covar, specific=specific)

    def read_with_date(self, dobj: Date, content: str) -> DataFrame: raise RuntimeError('should not hit this line')

    def serialize(self, factor_covar: DataFrame, spec_risk: Series, event_date: Date) -> None:
        """

        """
        # save factor covariance
        filename = self.filename.format(content='risk/factor_covar', event_date=event_date.strftime(self.date_format))
        mkdir(path=dirname(filename))
        factor_covar.to_parquet(filename, compression='gzip')
        # save spec risk
        filename = self.filename.format(content='risk/specific', event_date=event_date.strftime(self.date_format))
        mkdir(path=dirname(filename))
        spec_risk.to_frame(name='spec_risk').to_parquet(filename, compression='gzip')
        self.logger.info(f'saved risk model available for trading after bod of {event_date}')
