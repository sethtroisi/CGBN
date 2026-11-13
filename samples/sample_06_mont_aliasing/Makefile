ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

NVCCFLAGS ?= -O3 -std=c++14
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
	rm -f mont_aliasing

build:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=$(SM) mont_aliasing.cu -o mont_aliasing -lgmp

kepler:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_35 mont_aliasing.cu -o mont_aliasing -lgmp

maxwell:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_50 mont_aliasing.cu -o mont_aliasing -lgmp

pascal:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_60 mont_aliasing.cu -o mont_aliasing -lgmp

volta:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_70 mont_aliasing.cu -o mont_aliasing -lgmp

turing:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_75 mont_aliasing.cu -o mont_aliasing -lgmp

ampere:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_80 mont_aliasing.cu -o mont_aliasing -lgmp

ada:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_89 mont_aliasing.cu -o mont_aliasing -lgmp

hopper:
	nvcc $(NVCCFLAGS) $(INC) $(LIB) -I../../include -arch=sm_90 mont_aliasing.cu -o mont_aliasing -lgmp
