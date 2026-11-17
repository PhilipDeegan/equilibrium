#
# parsing scope funtion timers
#

import argparse
import sys
from dataclasses import dataclass

import phlop.timing.scope_timer as phst


@dataclass
class ScopeTimerFile(phst.ScopeTimerFile): ...


def file_parser(filepath):
    supe = phst.file_parser(filepath)
    return ScopeTimerFile(supe.id_keys, supe.roots)


def make_scope_timer_file_from(input):
    if type(input) is not list:
        return file_parser(None, input)
    supe = phst.lines_parser(input)
    return ScopeTimerFile(supe.id_keys, supe.roots, run=None)


def _cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default=None, help="timer file")
    parser.add_argument(
        "-F", "--filter", default=None, help="filter if function supports it"
    )
    return parser


def print_scope_timings(scope_timer_filepath=None, sort_worst_first=True, root_id=None):
    if scope_timer_filepath is None:  # assume cli
        parser = _cli_args()
        args = parser.parse_args()
        scope_timer_filepath = args.file
        if not scope_timer_filepath:
            parser.print_help()
            sys.exit(1)
        if args.filter:
            root_id = args.filter
    phst.print_scope_timings(scope_timer_filepath, sort_worst_first, root_id)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("usage: $function_name -h")
        print(
            "available functions:\n\t"
            + "\n\t".join([k for k, v in globals().items() if k.startswith("print_")]),
        )
    elif len(sys.argv) > 1:
        fn = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        if fn not in globals():
            raise ValueError("requested function does not exist")
        try:
            globals()[fn]()
        except BrokenPipeError:
            ...
