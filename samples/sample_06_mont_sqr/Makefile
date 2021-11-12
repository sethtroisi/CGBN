ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo "   make ampere"
	@echo

clean:
	rm -f mont_sqr

kepler:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 mont_sqr.cu -o mont_sqr -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 mont_sqr.cu -o mont_sqr -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_60 mont_sqr.cu -o mont_sqr -lgmp

volta:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 mont_sqr.cu -o mont_sqr -lgmp

turing:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 mont_sqr.cu -o mont_sqr -lgmp

ampere:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 mont_sqr.cu -o mont_sqr -lgmp

