# Copyright (c) 2021-2024 Patricio Cubillos
# puppies is open-source software under the GNU GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'msg',
    'warning',
    'error',
    'getpar',
    'cat',
    'Log',
]

import os
from pathlib import Path
import sys
import textwrap

import numpy as np
from mc3.utils import Log

import astropy.units as u


# Path to ROOT folder of the package:
ROOT = str(Path(__file__).parents[2]) + os.path.sep

# Warning/error banner:
sep = 70 * ":"


def log_error(self, message):
    """
    Pretty-print error message and end the code execution.

    Parameters
    ----------
    message: String
        String to be printed.
    tracklev: Integer
        Traceback level of error.
    """
    # Generate string to print:
    wrapped_text = self.wrap(message, indent=0)

    # Print to file, if requested:
    if self.file is not None and not self.file.closed:
        self.file.write(f"\n{self.sep}\n{wrapped_text}\n{self.sep}")
        self.close()

    # Close and exit:
    raise Exception(wrapped_text)

Log.error = log_error


def msg(verblevel, message, file=None, indent=0, si=None, noprint=False):
    """
    Conditional message printing to screen.

    Parameters
    ----------
    verblevel: Integer
        If positive, print the given message.
    message: String
        Message to print.
    file: File pointer
        If not None, print message to the given file pointer.
    indent: Integer
        Number of blank spaces for indentation.
    si: Integer
        Subsequent indentation.  If None, keep indent as the subsequent
        indentation.
    noprint: Boolean
        If True, do not print and return the string instead.
    """
    if verblevel < 0:
        return

    # Output text message:
    text = ""

    # Indentation strings:
    indspace = " "*indent
    sindspace = indspace
    if si is not None:
        sindspace = " "*si

    # Break the text down into sentences (line-breaks):
    sentences = message.splitlines()
    for s in sentences:
        line = textwrap.fill(
            s,
            break_long_words=False, break_on_hyphens=False,
            initial_indent=indspace, subsequent_indent=sindspace)
        text += line + "\n"

    # Do not print anywhere, just return the string:
    if noprint:
        return text

    # Print to screen:
    print(text[:-1])  # Remove the trailing "\n"
    sys.stdout.flush()
    # Print to file, if requested:
    if file is not None:
        file.write(text)
        file.flush()


def warning(verblevel, message, file=None):
    """
    Print message surrounded by colon banners.
    Append message to wlog.
    Add message to file if not None.

    Parameters
    ----------
    verblevel: Integer
       If positive, print the given message.
    message: String
       Message to print.
    file: File pointer
       If not None, also print to the given file.
    """
    if verblevel < 0:
      return

    # Wrap the message:
    text = msg(1, message, indent=4, noprint=True)[:-1]
    # Add banners around:
    warntext = "\n{:s}\n  Warning:\n{:s}\n{:s}\n".format(sep, text, sep)

    # Print warning message to screen:
    print(warntext)
    sys.stdout.flush()
    # Print warning message to file (if requested):
    if file is not None:
        file.write(warntext + "\n")
        file.flush()


def error(message, file=None):
    """
    Pretty print error message.

    Parameters
    ----------
    message: String
        Message to print.
    file: File pointer
        If not None, also print to the given file.
    """
    error_msg = msg(1, message, indent=0, noprint=True)

    # Print to file (if requested):
    if file is not None:
        file.write(f"\n{sep}\n{error_msg}\n{sep}")
        file.close()

    raise Exception(error_msg)


def getpar(par, units=u.dimensionless_unscaled, dtype=float):
    """
    Extract value, uncertainty, and units from input string.
    Apply dtype and unit casting as requested.

    Parameters
    ----------
    par: String
       A string with one to three blank-separated items: "value [[uncert] unit]"
    units: CompositeUnit
       An astropy's unit object.
    dtype: dtype
       The data type of the inputs.

    Returns
    -------
    value: Quantity
       The input value, casted to dtype, returned as the specified unit.
    error: Quantity
       The value's uncertainty (zero if there's no input error value).
    """
    val = par.split()

    # If input has more than one item, the last is the units:
    if len(val) > 1:
        units = u.Unit(val[-1])

    value = dtype(val[0]) * units

    # If input has three items, then the second is the uncertainty:
    if len(val) == 3:
        error = dtype(val[1]) * units
    else:
        error = 0 * units

    return value, error


# Short-hand for np.concatenate:
cat = np.concatenate
