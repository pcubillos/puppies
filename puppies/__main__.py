# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

import argparse
import webbrowser
import random

import puppies
from puppies.tools import ROOT


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
        help="Pick a random dog of the day, open it on the browser."
    )
    group.add_argument(
        "--today", dest='today', action='store_true',
        help="Show the the dog of the day on the browser."
    )
    group.add_argument(
        "-c", dest='cfile', default=None,
        help="Run puppies with given configuration file.",
    )

    # Parse command-line args:
    args, unknown = parser.parse_known_args()

    if args.today is True:
        webbrowser.open('https://dogperday.com/category/dog-of-the-day', new=2)
        return

    elif args.day is True:
        with open(f'{ROOT}puppies/data/dogs_of_the_day.txt', 'r') as f:
            dog_urls = f.readlines()
        ndogs = len(dog_urls)
        i = random.randint(1,ndogs)
        webbrowser.open(dog_urls[i], new=2)
        return


if __name__ == "__main__":
    main()

