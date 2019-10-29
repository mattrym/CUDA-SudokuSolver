#ifndef CPU_SOLVER_H_
#define CPu_SOLVER_H_

#include "sudoku.h"

typedef char FLAG;
typedef struct {
	FLAG progress;
	FLAG success;
	FLAG error;
} FLAGS;

void solve_cpu(BOARD, BOARD);

#endif