#ifndef CPU_SOLVER_H_
#define CPu_SOLVER_H_

#include "sudoku.h"

typedef char flag_t;
typedef struct {
	flag_t progress;
	flag_t success;
	flag_t error;
} flags_t;

void solve_cpu(board_t, board_t);

#endif