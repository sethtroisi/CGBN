NVCCFLAGS ?= -O3 -std=c++14
SM ?= sm_89

.PHONY: build clean

build:
	nvcc $(NVCCFLAGS) $(EXTRA) -I../../include -arch=$(SM) mont_sqr_edge.cu -o mont_sqr_edge -lgmp

clean:
	rm -f mont_sqr_edge
