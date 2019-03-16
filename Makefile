# Makefile for puppies project

# `make` - Build and compile the puppies executable and python extension.
# `make clean` - Remove all compiled (non-source) files that are created.

# If you are interested in the commands being run by this makefile, you may
# add "VERBOSE=1" to the end of any `make` command, i.e.:
#      make VERBOSE=1
# This will display the exact commands being used for building, etc.

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

LIBDIR = puppies/lib/


# Get the location of this Makefile.
mkfile_dir := $(dir $(lastword $(MAKEFILE_LIST)))

# `make [clean]` should run `make [clean]` on all of the modules.
all: make_pup make_mc3
#  make_eclipse
clean: clean_pup clean_mc3
# clean_eclipse


make_pup:
	@echo "Building puppies package."
	$(Q) python setup.py build $(O)
	@mv -f build/lib.*/puppies/lib/*.so $(LIBDIR)
	@rm -rf build/
	@echo "Successful compilation."
	@echo ""

make_mc3:
	@cd $(mkfile_dir)/modules/MCcubed/ && make

make_eclipse:
	@cd $(mkfile_dir)/modules/eclipse/ && make


clean_pup:
	@rm -rf $(LIBDIR)*.so
	@echo "Cleaned puppies."

clean_mc3:
	@cd $(mkfile_dir)/modules/MCcubed && make clean
	@echo "Cleaned MC3."

clean_eclipse:
	@cd $(mkfile_dir)/modules/eclipse && make clean
	@echo "Cleaned eclipse."

