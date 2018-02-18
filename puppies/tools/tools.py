import sys
import traceback
import textwrap
import numpy as np

import astropy.units  as u

__all__ = ["msg", "warning", "error", "getpar", "cat"]

# Warning/error banner:
sep = 70*":"


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
  indspace  = " "*indent
  sindspace = indspace
  if si is not None:
    sindspace = " "*si

  # Break the text down into sentences (line-breaks):
  sentences = message.splitlines()
  for s in sentences:
    line = textwrap.fill(s, break_long_words=False, break_on_hyphens=False,
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


def error(message, file=None, lev=-2):
  """
  Pretty print error message.

  Parameters
  ----------
  message: String
     Message to print.
  file: File pointer
     If not None, also print to the given file.
  """
  # Trace back the file, function, and line where the error source:
  t = traceback.extract_stack()
  # Extract fields:
  modpath    = t[lev][0]                       # Module path
  modname    = modpath[modpath.rfind('/')+1:]  # Madule name
  funcname   = t[lev][2]                       # Function name
  linenumber = t[lev][1]
  # Text to print:
  text = ("\n{:s}\n  Error in module: '{:s}', function: '{:s}', line: {:d}\n"
          "{:s}\n{:s}".format(sep, modname, funcname, linenumber,
                              msg(1,message,indent=4,noprint=True)[:-1], sep))

  # Print to screen:
  print(text)
  sys.stdout.flush()
  # Print to file (if requested):
  if file is not None:
    file.write(text)
    file.close()
  sys.exit(0)


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


def cat(a, b):
  """
  Short hand version of np.concatenate.

  Parameters
  ----------
  a, b: Sequences or ndarrays
     Arrays to concatenate. Must have compatible shapes, see
     help(np.concatenate)
  """
  return np.concatenate([a,b])
