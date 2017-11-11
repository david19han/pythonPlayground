#
# This Makefile should be used as a template for future
# Makefiles.  It's heavily commented, so hopefully you can
# understand what each line does.  If anything is unclear,
# please ask in the mailing list, in office hours, or in
# class.
#

# We'll use gcc for C compilation and g++ for C++ compilation

CC  = gcc
CXX = g++

# Let's leave a place holder for additional include directories

INCLUDES =

# Compilation options:
# -g for debugging info and -Wall enables all warnings

CFLAGS   = -g -Wall $(INCLUDES)
CXXFLAGS = -g -Wall $(INCLUDES)

# Linking options:
# -g for debugging info

LDFLAGS = -g

# List the libraries you need to link with in LDLIBS
# For example, use "-lm" for the math library
# if you used sqrt() function.

LDLIBS = -lm

# The 1st target gets built when you type "make".
# It's usually your executable.  ("main" in this case.)
#
# Note that we did not specify the linking rule.
# Instead, we rely on one of make's implicit rules:
#
#     $(CC) $(LDFLAGS) <all-dependent-.o-files> $(LDLIBS)
#
# Also note that make assumes that main depends on main.o,
# so we can omit it.

.PHONY: runcl
runcl:
	sbatch --gres=gpu:1 --time=5 --wrap="python 2DConvCL.py"
.PHONY: runcl2

.PHONY: runcuda
runcuda:
	sbatch --gres=gpu:1 --time=5 --wrap="python 2DConvCUDA.py"

.PHONY: clean
clean:
	rm -f *.o a.out core slurm*
