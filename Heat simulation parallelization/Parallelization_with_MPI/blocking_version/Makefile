CC =  mpicc
CFLAGS = -O3 -xhost $(PAPI_INC) 

MPICC = mpicc

all: heat 

heat : heat.o input.o misc.o timing.o relax_jacobi.o
	$(MPICC) $(CFLAGS) -o heat $+ -lm  $(PAPI_LIB)

%.o : %.c %.h
	$(MPICC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o heat *~ *.ppm

remake : clean all
