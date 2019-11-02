#ifndef SUDOKU_CUH_
#define SUDOKU_CUH_

#define n 3
#define N 9
#define BOARD_SIZE 81

#define ROW_MASK 0
#define COL_MASK 1
#define SUB_MASK 2

#define FULL_MASK 511

typedef __int8 cell_t;
typedef cell_t* board_t;

typedef __int16 mask_t;
typedef __int16 candidates_t;
typedef board_t boards_t;

#endif