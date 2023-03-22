#ifndef JACOBI_H_INCLUDED
#define JACOBI_H_INCLUDED

#include <stdio.h>
#include <mpi.h>

// configuration

typedef struct
{
    float posx;
    float posy;
    float range;
    float temp;
}
heatsrc_t;

typedef struct
{
    unsigned maxiter;       // maximum number of iterations
    unsigned act_res;
    unsigned max_res;       // spatial resolution
    unsigned initial_res;
    unsigned res_step_size;
    unsigned visres;        // visualization resolution

    //Added variables
    int startIndexX;
    int endIndexX;
    int sizeX;
    int startIndexY;
    int endIndexY;
    int sizeY;
  
    double *u, *uhelp;
    double *uvis;

    unsigned   numsrcs;     // number of heat sources
    heatsrc_t *heatsrcs;
}
algoparam_t;


// function declarations

// misc.c
int initialize( algoparam_t *param );
int finalize( algoparam_t *param );
void write_image( FILE * f, double *u,
		  unsigned sizex, unsigned sizey );
int coarsen( double *uold, unsigned oldx, unsigned oldy, 
			unsigned startIndexX, unsigned startIndexY, unsigned npX, unsigned npY,
	    	double *unew, unsigned newx, unsigned newy);

// Gauss-Seidel: relax_gauss.c
double residual_gauss( double *u, double *utmp,
		       unsigned sizex, unsigned sizey );
void relax_gauss( double *u, 
		  unsigned sizex, unsigned sizey  );

// Jacobi: relax_jacobi.c
double residual_jacobi( double *u,
			unsigned sizex, unsigned sizey );
double relax_jacobi( double **u1, double **utmp1, unsigned sizex, unsigned sizey, int neighbours_ranks[], 
            MPI_Comm * comm2d, double **send_columnL1, double **send_columnR1, double **recv_columnL1, double **recv_columnR1);


#endif // JACOBI_H_INCLUDED
