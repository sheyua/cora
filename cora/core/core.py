from typing import List
from ..util.config import Config
from ..util.config.parser import command_line_override
from ..util.dynamic import load
from ..util.logger import make_logger
from .state import State


def build_config(filename: str, override: List[str]) -> Config:
    """
        make config first, config is the 'command line override' + 'user config' + 'system default'
    """
    # do not parse override before absorbing the default, user might override on the default
    ans = Config.make_config(filename=filename, override=None)
    ans.override(other=command_line_override(override=override))

    return ans


def do_core(filename: str, override: List[str]) -> None:
    """

    """
    # make config and logger given the config
    config = build_config(filename=filename, override=override)
    logger = make_logger(config=config, name=__name__)

    # make a state
    state = State(config=config, logger=logger)
    State.set(state=state)
    try:
        for name in state.root_nodes:
            padding = '=' * 10
            state.logger.info(f'{padding}{name:^20}{padding}')
            module_name = config.sub_root(section=name).as_str(key='module_name')
            class_name = config.sub_root(section=name).as_str(key='class_name', default=name)
            class_ = load(module_name=module_name, class_name=class_name)
            object_ = class_(name=name)
            object_.run()

    except Exception as err:
        from traceback import format_exc
        tb = format_exc()
        logger.critical(f'failed due to unexpected err {err}, {tb}')
        raise err
