# Makefile for puppies project

# `make` - Build and compile the puppies executable and python extension.
# `make clean` - Remove all compiled (non-source) files that are created.

# If you are interested in the commands being run by this makefile, you may
# add "VERBOSE=1" to the end of any `make` command, i.e.:
#      make VERBOSE=1
# This will display the exact commands being used for building, etc.

# To enforce compilation with Python3 append "PY3=1" to the make command:
#      make PY3=1

LIBDIR = puppies/lib/

# Set verbosity
#
Q = @
O = > /dev/null

ifdef VERBOSE
	ifeq ("$(origin VERBOSE)", "command line")
		Q =
		O =
	endif
else
	MAKEFLAGS += --no-print-directory
endif


# Get the location of this Makefile.
mkfile_dir := $(dir $(lastword $(MAKEFILE_LIST)))

# `make [clean]` should run `make [clean]` on all of the modules.
all: make_pup
clean: clean_pup


make_pup:
	@echo "Building puppies package."
	$(Q) python setup.py build $(O)
	@mv -f build/lib.*/puppies/lib/*.so $(LIBDIR)
	@rm -rf build/
	@echo "Successful compilation."
	@echo ""

clean_pup:
	@rm -rf $(LIBDIR)*.so
	@echo "Cleaned puppies."

