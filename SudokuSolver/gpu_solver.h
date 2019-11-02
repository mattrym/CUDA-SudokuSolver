#include "sudoku.h"

#ifndef GPU_SOLVER_H_
#define GPU_SOLVER_H_

#define BLOCKS 65535
#define THREADS 27
#define ITERATIONS 100

#define IDLE 0
#define BUSY 1
#define SUCCESS 2

typedef char block_flag_t;
typedef int block_status_t;

typedef struct {
	block_flag_t progress;
	block_flag_t success;
	block_flag_t error;
} block_flags_t;

void solve_gpu(board_t, board_t);

#endif