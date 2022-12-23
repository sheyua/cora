from typing import List
from logging import Logger, WARNING, ERROR
from ..util.constant import DEFAULT_LOGGING_FORMATTER, DEFAULT_MAX_SIZE, DEFAULT_TIMEZONE
from ..util.config import Config, OverrideType
from ..util.logger import LevelType, make_console


class State(object):
    """
        state is the overall resource manager
    """
    _instance = None

    def __init__(self, config: Config, logger: Logger) -> None:
        """

        """
        if not config.is_root:
            raise ValueError('can only init state with root config')
        self.config = config
        self.logger = logger

    @property
    def root_nodes(self) -> List[str]:
        """

        """
        return self.config.get_list(section='State', option='main', default='')

    @staticmethod
    def set(state: 'State') -> None:
        """

        """
        if not isinstance(state, State):
            raise TypeError('input is not a state')
        State._instance = state

    @staticmethod
    def get() -> 'State':
        """

        """
        if isinstance(State._instance, State):
            return State._instance
        else:
            return State.console_state()

    @staticmethod
    def reset() -> None:
        """

        """
        if isinstance(State._instance, State):
            State._instance = None

    @staticmethod
    def console_state(filename: str='', override: OverrideType=None, stdout: LevelType=WARNING, stderr: LevelType=ERROR,
                      formatter: str=DEFAULT_LOGGING_FORMATTER) -> 'State':
        """

        """
        if isinstance(State._instance, State):
            return State._instance
        else:
            config = Config.make_config(filename=filename, override=override)
            logger = make_console(name=__name__, stdout=stdout, stderr=stderr, reset=True, formatter=formatter)
            ans = State(config=config, logger=logger)
            State._instance = ans
            return ans
