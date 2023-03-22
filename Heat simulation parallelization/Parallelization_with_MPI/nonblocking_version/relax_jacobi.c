/*
 * relax_jacobi.c
 *
 * Jacobi Relaxation
 *
 */

#include "heat.h"
#include <mpi.h>
#include <stdlib.h>

double relax_jacobi( double **u1, double **utmp1, unsigned sizex, unsigned sizey, int neighbours_ranks[], MPI_Comm * comm2d, double **send_columnL1, double **send_columnR1, double **recv_columnL1, double **recv_columnR1){
  //create send request (not important to be different)
  MPI_Request reqsend;
  //create 4 recv requests for each side
	MPI_Request reqrec0;
	MPI_Request reqrec1;
	MPI_Request reqrec2;
	MPI_Request reqrec3;

  MPI_Comm comm_2d = *comm2d;

  int i, j;
  double *help, *u, *utmp, *send_columnL, *send_columnR, *recv_columnL, *recv_columnR, factor=0.5;
  MPI_Status s;
  enum DIRECTIONS {DOWN, UP, LEFT, RIGHT};

  utmp = *utmp1;
  u = *u1;
  send_columnL = *send_columnL1;
  send_columnR = *send_columnR1;
  recv_columnL = *recv_columnL1;
  recv_columnR = *recv_columnR1;
  

  /* 1. Sending and receiving ghost layers for u */
  if (neighbours_ranks[DOWN] != MPI_PROC_NULL){
    //Sending
    MPI_Isend(u + (sizey-2) * sizex + 1, (sizex - 2), MPI_DOUBLE, neighbours_ranks[DOWN], 0, comm_2d, &reqsend);

    //Receiving
    MPI_Irecv(u + (sizey-1) * sizex + 1, (sizex - 2), MPI_DOUBLE, neighbours_ranks[DOWN], 1, comm_2d, &reqrec0);
  }	
  if (neighbours_ranks[UP] != MPI_PROC_NULL){
    //Sending
    MPI_Isend(u + sizex + 1, (sizex - 2), MPI_DOUBLE, neighbours_ranks[UP], 1, comm_2d, &reqsend);

    //Receiving
    MPI_Irecv(u + 1, (sizex - 2), MPI_DOUBLE, neighbours_ranks[UP], 0, comm_2d, &reqrec1);
  }
  
  if (neighbours_ranks[RIGHT] != MPI_PROC_NULL){
    for (int i = 1; i <= sizey - 2; i++){
      send_columnR[i-1] = u[(i+1)*sizex-2];
    }

    //Sending
    MPI_Isend(send_columnR, (sizey - 2), MPI_DOUBLE, neighbours_ranks[RIGHT], 2, comm_2d, &reqsend);
  }
  if (neighbours_ranks[LEFT] != MPI_PROC_NULL){
    for (int i = 1; i <= sizey - 2; i++){
      send_columnL[i-1] = u[i*sizex +1];
    }
    //Sending
    MPI_Isend(send_columnL, (sizey - 2), MPI_DOUBLE, neighbours_ranks[LEFT], 3, comm_2d, &reqsend);
  }

  if (neighbours_ranks[RIGHT] != MPI_PROC_NULL){
    //Receiving
    MPI_Irecv(recv_columnR, sizey - 2, MPI_DOUBLE, neighbours_ranks[RIGHT], 3, comm_2d, &reqrec2);
  }

  if (neighbours_ranks[LEFT] != MPI_PROC_NULL){
    //Receiving
    MPI_Irecv(recv_columnL, sizey - 2, MPI_DOUBLE, neighbours_ranks[LEFT], 2, comm_2d, &reqrec3);
  }


/* 2. Calculating the center portion which doesn't need to wait for communication to be over */
  double unew, diff, sum = 0.0;
  int ii, iim1, iip1;
  for( i=2; i<sizey-2; i++ ) {
  	ii=i*sizex;
  	iim1=(i-1)*sizex;
  	iip1=(i+1)*sizex;
#pragma ivdep
    for( j=2; j<sizex-2; j++ ){
      unew = 0.25 * (u[ ii+(j-1) ]+
                      u[ ii+(j+1) ]+
                      u[ iim1+j ]+
                      u[ iip1+j ]);
      diff = unew - u[ii + j];
      utmp[ii+j] = unew;
      sum += diff * diff;

      }
    }


/* 3. Calculating the outer layers */
//calculating the vertical boundary layers after receiving the ghost layers
if (neighbours_ranks[LEFT] != MPI_PROC_NULL){
  MPI_Wait(&reqrec3, &s); //waiting for left side to arrive
  for (int i = 1; i <= sizey - 2; i++){
    u[i*sizex] = recv_columnL[i-1];
  }
}
#pragma ivdep 
  for (i = 2; i < sizey - 2; i++){
    ii = i * sizex;
    iim1 = (i - 1) * sizex;
    iip1 = (i + 1) * sizex;

    //using left layer
      unew = 0.25 * (u[ii] +
                      u[ii + 2] +
                      u[iim1 + 1] +
                      u[iip1 + 1]);
      diff = unew - u[ii + 1];
      utmp[ii + 1] = unew;
      sum += diff * diff;
  }

if (neighbours_ranks[RIGHT] != MPI_PROC_NULL){
  MPI_Wait(&reqrec2, &s); //waiting for right side to arrive
  for (int i = 1; i <= sizey - 2; i++){
    u[i*sizex + sizex - 1] = recv_columnR[i-1];
  }
}
#pragma ivdep 
  for (i = 2; i < sizey - 2; i++){
    ii = i * sizex;
    iim1 = (i - 1) * sizex;
    iip1 = (i + 1) * sizex;

    //using right layer
    j = sizex-2;
      unew = 0.25 * (u[ii + (j - 1)] +
                      u[ii + (j + 1)] +
                      u[iim1 + j] +
                      u[iip1 + j]);
      diff = unew - u[ii + j];
      utmp[ii + j] = unew;
      sum += diff * diff;
  }

//calculating the vertical horizontal layers after receiving the ghost layers
if (neighbours_ranks[UP] != MPI_PROC_NULL){
  MPI_Wait(&reqrec1, &s); //wait for top layer to arrive
}
#pragma ivdep
  for (j = 1; j < sizex - 1; j++){
    //Using top layer
    ii = sizex;
    iim1 = 0;
    iip1 = 2 * sizex;
    unew = 0.25 * (u[ii + (j - 1)] +
                    u[ii + (j + 1)] +
                    u[iim1 + j] +
                    u[iip1 + j]);
    diff = unew - u[ii + j];
    utmp[ii + j] = unew;
    sum += diff * diff;
  }

if (neighbours_ranks[DOWN] != MPI_PROC_NULL){
  MPI_Wait(&reqrec0, &s); //wait for bottom layer to arrive
}
#pragma ivdep    
  for (j = 1; j < sizex - 1; j++){
    //Using down layer
    i = sizey - 2;
    ii = i * sizex;
    iim1 = (i - 1) * sizex;
    iip1 = (i + 1) * sizex;
    unew = 0.25 * (u[ii + (j - 1)] +
                    u[ii + (j + 1)] +
                    u[iim1 + j] +
                    u[iip1 + j]);
    diff = unew - u[ii + j];
    utmp[ii + j] = unew;
    sum += diff * diff;
  }

/* 4. Swapping the arrays and returning the residual */
  *u1 = utmp;
  *utmp1 = u;
  return (sum);
}