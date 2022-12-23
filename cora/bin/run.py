from sys import path, argv
from argparse import ArgumentParser
from os.path import dirname, abspath


def make_parser() -> ArgumentParser:
    """
        parse command line information
    """
    ans = ArgumentParser(description='main executable')
    ans.add_argument('-c', '--config', dest='config', default='yaml/test.yaml', help='provide a config file to start')
    ans.add_argument('--profile', dest='profile', help='location of an optional profiling graph')
    ans.add_argument('--profile-max-depth', dest='profile_max_depth', type=int, help='maximum depth for profiling')
    ans.add_argument('positional', nargs='*')
    return ans


def main() -> None:
    """
        main function
    """
    args = make_parser().parse_args(argv[1:])
    if not args.config:
        raise RuntimeError('must specify a config')

    # do_core
    from cora.core.core import do_core

    if args.profile is not None:
        from pycallgraph import PyCallGraph, GlobbingFilter, Config as CallGraphConfig
        from pycallgraph.output import GraphvizOutput
        from exciton.util.file import package_name

        if args.profile_max_depth is not None:
            profile_config = CallGraphConfig(max_depth=args.profile_max_depth)
        else:
            profile_config = CallGraphConfig()
        profile_config.trace_filter = GlobbingFilter(include=[f'{package_name()}.*'])
        graph = GraphvizOutput()
        graph.output_file = args.profile
        with PyCallGraph(output=graph, config=profile_config):
            do_core(filename=args.config, override=args.positional)
    else:
        do_core(filename=args.config, override=args.positional)


if __name__ == '__main__':
    project_directory = abspath(dirname(__file__) + '/../..')
    if project_directory not in path:
        path.append(project_directory)
    try:
        main()
    except KeyboardInterrupt:
        print('ctrl + c pressed')
        exit(1)
