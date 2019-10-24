#ifndef SOLVER_CUH_
#define SOLVER_CUH_

#include <cuda_runtime.h>

#define BLOCKS 65535
#define THREADS 27
#define ITERATIONS 100

#define ROW_MASK 0
#define COL_MASK 1
#define SUB_MASK 2

#define IDLE 0
#define BUSY 1
#define SUCCESS 2

#define FULL_MASK 511

typedef char BLOCK_FLAG;
typedef int BLOCK_STATUS;

typedef __int16 MASK;
typedef __int16 CANDIDATES;
typedef __int8* BOARDS;

typedef struct {
	BLOCK_FLAG progress;
	BLOCK_FLAG success;
	BLOCK_FLAG error;
} BLOCK_FLAGS;

cudaError_t run_solve(BOARD, BOARD, int*);

#endif