ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

NVCCFLAGS ?= -O3 -std=c++14 -Xcompiler -pthread
SM ?= sm_80

.PHONY: pick clean build kepler maxwell pascal volta turing ampere ada hopper

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo "   make ampere"
	@echo "   make ada"
	@echo "   make hopper"
	@echo "   make build SM=sm_XX"
	@echo

clean:
	rm -f mont_sqr

build:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=$(SM) mont_sqr.cu -o mont_sqr -lgmp -lpthread

kepler:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_35 mont_sqr.cu -o mont_sqr -lgmp -lpthread

maxwell:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_50 mont_sqr.cu -o mont_sqr -lgmp -lpthread

pascal:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_60 mont_sqr.cu -o mont_sqr -lgmp -lpthread

volta:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_70 mont_sqr.cu -o mont_sqr -lgmp -lpthread

turing:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_75 mont_sqr.cu -o mont_sqr -lgmp -lpthread

ampere:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_80 mont_sqr.cu -o mont_sqr -lgmp -lpthread

ada:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_89 mont_sqr.cu -o mont_sqr -lgmp -lpthread

hopper:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_90 mont_sqr.cu -o mont_sqr -lgmp -lpthread
