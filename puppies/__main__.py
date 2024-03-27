# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

import argparse
import webbrowser

import puppies


def main():
    """
    puppies: The Public Photometry Pipeline for Exoplanets
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version", action="version",
        help="Show puppies's version.",
        version=f'puppies version {puppies.__version__}.'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-d", "--day", dest='day', action='store_true',
        help="Show the the dog of the day on the browser."
    )
    group.add_argument(
        "-c", dest='cfile', default=None,
        help="Run puppies with given configuration file.",
    )

    # Parse command-line args:
    args, unknown = parser.parse_known_args()

    # Parse configuration file to a dictionary:
    if args.day is True:
        webbrowser.open('https://dogperday.com/category/dog-of-the-day', new=2)
        return


if __name__ == "__main__":
    main()

