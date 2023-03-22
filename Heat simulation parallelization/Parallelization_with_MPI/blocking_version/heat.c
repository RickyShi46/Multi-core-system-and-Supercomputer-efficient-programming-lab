#include <stdio.h>

#include "input.h"
#include "heat.h"
#include "timing.h"
#include "omp.h"
#include "mmintrin.h"
#include <mpi.h>
#include <papi.h>
#include <string.h>

double* time;

double gettime() {
	return ((double) PAPI_get_virt_usec() * 1000000.0);
}

void usage(char *s) {
	fprintf(stderr, "Usage: %s <input file> [result file]\n\n", s);
}

int main(int argc, char *argv[]) {
	int i, j, k, ret;
	FILE *infile, *resfile;
	char *resfilename;
	int np, iter, chkflag;
	double rnorm0, rnorm1, t0, t1;
	double tmp[8000000];

	double totalResidual = 0;
	//MPI things
	int nTaskx = 3; //initialize with 3 tasks in x direction 2 in y direction, but the flags will change it
    int nTasky = 2;
	//Reading flag values
	if (argc >= 5)
	{
		if (strcmp(argv[2], "-x") == 0)
			nTaskx = atoi(argv[3]);
		if (strcmp(argv[4], "-x") == 0)
			nTaskx = atoi(argv[5]);
		if (strcmp(argv[2], "-y") == 0)
			nTasky = atoi(argv[3]);
		if (strcmp(argv[4], "-y") == 0)
			nTasky = atoi(argv[5]);
	}
    int dims[2] = {nTaskx, nTasky};
    int size, rank;
    int * coords; //the 2 dimensional coordinate of the rank
    coords = calloc(2, sizeof(int));
    int periods[2] = {0,0}; //no periods
    int reorder = 1; //reorder true

    MPI_Status s;
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	if (rank==0) printf("%d number of processes have started\n", size);
	if (rank==0) printf("%d, %d, number of x, y, processes\n", nTaskx, nTasky);

    //creating a 2d topology
    MPI_Comm comm_2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm_2d); //creating nTaskx x nTasky no periodic ordering = true
    MPI_Cart_coords(comm_2d, rank, 2, coords); //loading their own coordinate

    int newRank;
    MPI_Comm_rank(comm_2d, &newRank); //getting the new rank

	//figuring out the ranks of neighbors
	enum DIRECTIONS {DOWN, UP, LEFT, RIGHT};
	char* neighbours_names[4] = {"down", "up", "left", "right"};
	int neighbours_ranks[4];

	// Let consider dims[0] = X, so the shift tells us our left and right neighbours
	MPI_Cart_shift(comm_2d, 0, 1, &neighbours_ranks[LEFT], &neighbours_ranks[RIGHT]);

	// Let consider dims[1] = Y, so the shift tells us our up and down neighbours
	MPI_Cart_shift(comm_2d, 1, 1, &neighbours_ranks[UP], &neighbours_ranks[DOWN]);

	// algorithmic parameters
	algoparam_t param;

	double residual;

	// set the visualization resolution
	param.visres = 100;

	// check arguments
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	// check input file
	if (!(infile = fopen(argv[1], "r"))) {
		fprintf(stderr, "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);

		usage(argv[0]);
		return 1;
	}
	MPI_Barrier(MPI_COMM_WORLD);

	// check result file
	resfilename = "heat.ppm";

	if (!(resfile = fopen(resfilename, "w"))) {
		fprintf(stderr, "\nError: Cannot open \"%s\" for writing.\n\n", resfilename);

		usage(argv[0]);
		return 1;
	}
	MPI_Barrier(MPI_COMM_WORLD);

	// check input
	if (!read_input(infile, &param)) {
		fprintf(stderr, "\nError: Error parsing input file.\n\n");

		usage(argv[0]);
		return 1;
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) print_params(&param);
	MPI_Barrier(MPI_COMM_WORLD);
	time = (double *) calloc(sizeof(double), (int) (param.max_res - param.initial_res + param.res_step_size) / param.res_step_size);
	MPI_Barrier(MPI_COMM_WORLD);
	int exp_number = 0;

	for (param.act_res = param.initial_res; param.act_res <= param.max_res; param.act_res = param.act_res + param.res_step_size) {
		//calculate which part of the global array belongs to this process
		param.startIndexX = (coords[0] * param.act_res) / nTaskx; 				//inclusive
		param.endIndexX = ((coords[0] + 1) * param.act_res) / nTaskx; 			//not inclusive
		param.sizeX = param.endIndexX - param.startIndexX; 						//doesn't include ghost +2
		param.startIndexY = (coords[1] * param.act_res) / nTasky; 				//inclusive
		param.endIndexY = ((coords[1] + 1) * param.act_res) / nTasky; 			//not inclusive
		param.sizeY = param.endIndexY - param.startIndexY; 						//doesn't include ghost +2

		if (!initialize(&param)) {
			fprintf(stderr, "Error in Jacobi initialization.\n\n");

			usage(argv[0]);
		}
		//printf("startIndexX = %d, endIndexX = %d, startIndexY= %d, endIndexY= %d, sizeX= %d, sizeY= %d\n", param.startIndexX, param.endIndexX, param.startIndexY, param.endIndexY, param.sizeX, param.sizeY);
		
		//copying param.u to param.uhelp
		for (i = 0; i < param.sizeX + 2; i++) {
			for (j = 0; j < param.sizeY + 2; j++) {
				param.uhelp[i * (param.sizeY + 2) + j] = param.u[i * (param.sizeY + 2) + j];
			}
		}

		// starting time
		time[exp_number] = wtime();
		residual = 999999999;
		np = param.act_res + 2;

		t0 = gettime();

		//ghost layer sizes and allocating send and recv columns
		const int npX = param.sizeX + 2;
		const int npY = param.sizeY + 2;
		double* send_columnL = (double*)malloc( sizeof(double)* param.sizeY);
		double* recv_columnL = (double*)malloc( sizeof(double)* param.sizeY);
		double* send_columnR = (double*)malloc( sizeof(double)* param.sizeY);
		double* recv_columnR = (double*)malloc( sizeof(double)* param.sizeY);


		for (iter = 0; iter < param.maxiter; iter++) {
			
			//Sending and receiving ghost layers for u
			if (neighbours_ranks[DOWN] != MPI_PROC_NULL){
				//Sending
				MPI_Send(param.u + (npY-2) * npX + 1, param.sizeX, MPI_DOUBLE, neighbours_ranks[DOWN], 0, comm_2d);

				//Receiving
				MPI_Recv(param.u + (npY-1) * npX + 1, param.sizeX, MPI_DOUBLE, neighbours_ranks[DOWN], 1, comm_2d, &s);
			}	
			if (neighbours_ranks[UP] != MPI_PROC_NULL){
				//Sending
				MPI_Send(param.u + npX + 1, param.sizeX, MPI_DOUBLE, neighbours_ranks[UP], 1, comm_2d);

				//Receiving
				MPI_Recv(param.u + 1, param.sizeX, MPI_DOUBLE, neighbours_ranks[UP], 0, comm_2d, &s);
			}
			
			if (neighbours_ranks[RIGHT] != MPI_PROC_NULL){
				for (int i = 1; i <= npY - 2; i++){
					send_columnR[i-1] = param.u[(i+1)*npX-2];
				}
				//Sending
				MPI_Send(send_columnR, param.sizeY, MPI_DOUBLE, neighbours_ranks[RIGHT], 2, comm_2d);

				//Receiving
				MPI_Recv(recv_columnR, param.sizeY, MPI_DOUBLE, neighbours_ranks[RIGHT], 3, comm_2d, &s);
				for (int i = 1; i <= npY - 2; i++){
					param.u[i*npX + npX - 1] = recv_columnR[i-1];
				}
			}
			if (neighbours_ranks[LEFT] != MPI_PROC_NULL){
				for (int i = 1; i <= npY - 2; i++){
					send_columnL[i-1] = param.u[i*npX +1];
				}
				//Sending
				MPI_Send(send_columnL, param.sizeY, MPI_DOUBLE, neighbours_ranks[LEFT], 3, comm_2d);

				//Receiving
				MPI_Recv(recv_columnL, param.sizeY, MPI_DOUBLE, neighbours_ranks[LEFT], 2, comm_2d, &s);
				for (int i = 1; i <= npY - 2; i++){
					param.u[i*npX] = recv_columnL[i-1];
				}
			}

			//relax jacobi 1 step
			residual = relax_jacobi(&(param.u), &(param.uhelp), npX, npY);

		}

		//Reduce the residual
		MPI_Reduce(&residual, &totalResidual, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		t1 = gettime();
		time[exp_number] = wtime() - time[exp_number];
		//printf("rank = %d, residual = %f\n", newRank, residual);
		MPI_Barrier(MPI_COMM_WORLD);

		if(rank == 0){
			printf("\n\nResolution: %u\n", param.act_res);
			printf("===================\n");
			printf("Execution time: %f\n", time[exp_number]);
			printf("Residual: %f\n\n", totalResidual);

			printf("megaflops:  %.1lf\n", (double) param.maxiter * (np - 2) * (np - 2) * 7 / time[exp_number] / 1000000);
			printf("  flop instructions (M):  %.3lf\n", (double) param.maxiter * (np - 2) * (np - 2) * 7 / 1000000);
		}

		exp_number++;

		/*
		//printing out one by one
		int printRank = 0;
		while (printRank < size) {
		if (newRank == printRank) {
			printf("old rank: %d, new rank: %d, Coords: [%d,%d], x-range: %d-%d, sizeX = %d, y-range: %d-%d, sizeY = %d\n", rank, newRank, coords[0], coords[1], param.startIndexX, param.endIndexX, param.sizeX, param.startIndexY, param.endIndexY, param.sizeY);
			for(int j = 0; j < param.sizeY+2; j++){
				for(int i = 0; i < param.sizeX+2; i++){
					printf("%f ", (param.u)[j*(param.sizeX+2) + i]);
				}
				printf("\n");
			}
		}
		printRank++;
		MPI_Barrier (MPI_COMM_WORLD);
		}
		*/
		
	}

	param.act_res = param.act_res - param.res_step_size;

	//coarsen the array to visualize
	coarsen(param.u, param.act_res + 2, param.act_res + 2, param.startIndexX, param.startIndexY, param.sizeX + 2, param.sizeY + 2, param.uvis, param.visres + 2, param.visres + 2);
	MPI_Barrier(MPI_COMM_WORLD);

	//reduce the coarsened arrays to the root
	double * uvisGlobal;
	if(rank==0) uvisGlobal = (double*)calloc( sizeof(double),
				      (param.visres+2) *
				      (param.visres+2) );
	MPI_Reduce(param.uvis, uvisGlobal, (param.visres + 2)*(param.visres + 2), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	//root creates the image
	if(rank==0) write_image(resfile, uvisGlobal, param.visres + 2, param.visres + 2);
	if(rank==0) free(uvisGlobal);

	//finalize
	finalize(&param);
	MPI_Finalize();
	return 0;
}