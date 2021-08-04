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
	rm -f ecm_s1

kepler:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 ecm_s1.cu -o ecm_s1 -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 ecm_s1.cu -o ecm_s1 -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_61 ecm_s1.cu -o ecm_s1 -lgmp

volta:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 ecm_s1.cu -o ecm_s1 -lgmp

turing:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 ecm_s1.cu -o ecm_s1 -lgmp

ampere:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 ecm_s1.cu -o ecm_s1 -lgmp

