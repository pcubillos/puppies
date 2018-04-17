# Makefile - prepared for puppies
#
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


DIRECTIVE = 
ifdef PY3
	ifeq ("$(origin PY3)", "command line")
		DIRECTIVE = 3
	endif
endif


# Get the location of this Makefile.
mkfile_dir := $(dir $(lastword $(MAKEFILE_LIST)))

# `make [clean]` should run `make [clean]` on all of the modules.
all: make_pup make_mc3 make_eclipse
clean: clean_pup clean_mc3 clean_eclipse


make_pup:
	@echo "Building puppies package."
	$(Q) python$(DIRECTIVE) setup.py build $(O)
	@mv -f build/lib.*/*.so $(LIBDIR)
	@rm -rf build/
	@echo "Successful compilation.\n"

make_mc3:
	@cd $(mkfile_dir)/modules/MCcubed/ && make

make_eclipse:
	@cd $(mkfile_dir)/modules/eclipse/ && make


clean_pup:
	@rm -rf $(LIBDIR)*.so
	@echo "Cleaned Pyrat Bay.\n"

clean_mc3:
	@cd $(mkfile_dir)/modules/MCcubed && make clean
	@echo "Cleaned MC3.\n"

clean_eclipse:
	@cd $(mkfile_dir)/modules/eclipse && make clean
	@echo "Cleaned eclipse.\n"

