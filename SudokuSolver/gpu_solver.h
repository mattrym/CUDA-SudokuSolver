#include "sudoku.h"

#ifndef GPU_SOLVER_H_
#define GPU_SOLVER_H_

#define BLOCKS 65535
#define THREADS 27
#define ITERATIONS 100

#define IDLE 0
#define BUSY 1
#define SUCCESS 2

typedef char BLOCK_FLAG;
typedef int BLOCK_STATUS;

typedef struct {
	BLOCK_FLAG progress;
	BLOCK_FLAG success;
	BLOCK_FLAG error;
} BLOCK_FLAGS;

void solve_gpu(BOARD, BOARD);

#endif