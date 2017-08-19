import sys
import traceback
import textwrap
import numpy as np

__all__ = ["parray", "msg", "warning", "error"]

# Warning/error banner:
sep = 70*":"


def parray(string, dtype=None):
  """
  Convert a string containin a list of white-space-separated (and/or
  newline-separated) values into a numpy array
  """
  if string == 'None':
    return [None]
  # Return list of strings if dtype is None:
  if dtype is None:
    return string.split()
  # Else, convert to numpy array if given type
  return np.asarray(string.split(), dtype)


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

  # Do not print, just return the string:
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

