include make.inc

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
        MPIFLAGS += -stdlib=libstdc++ -lstdc++
endif

all: bin/al3c 

static_darwin: bin/al3c_darwin_x86-64

static_linux: bin/al3c_linux_x86-64

bin/al3c:  src/main.cpp src/SMC.cpp src/signal.cpp  src/u01.cpp  src/weight.cpp  include/al3c.hpp make.inc
	$(MPI) $(MPIFLAGS) $(MPIINCLUDEFLAGS)  -o bin/al3c  $(MPILIBFLAGS) $< -ldl

##make sure MPI .dylibs are removed for this to compile statically...
bin/al3c_darwin_x86-64:  src/main.cpp src/SMC.cpp src/signal.cpp  src/u01.cpp  src/weight.cpp  include/al3c.hpp make.inc
	$(MPI) $(MPIFLAGS) $(MPIINCLUDEFLAGS)  -o bin/al3c_darwin_x86-64  $(MPILIBFLAGS) $<

bin/al3c_linux_x86-64: src/main.cpp src/SMC.cpp src/signal.cpp  src/u01.cpp  src/weight.cpp  include/al3c.hpp make.inc
	mpiicpc -Wall -O2 -static_mpi -o bin/al3c_linux_x86-64 -ldl  $<

clean: 
	-rm bin/al3c bin/al3c_linux_x86-64
