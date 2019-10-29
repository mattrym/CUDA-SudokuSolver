#ifndef SUDOKU_CUH_
#define SUDOKU_CUH_

#define n 3
#define N 9
#define BOARD_SIZE 81

#define ROW_MASK 0
#define COL_MASK 1
#define SUB_MASK 2

#define FULL_MASK 511

typedef __int8 CELL;
typedef CELL* BOARD;

typedef __int16 MASK;
typedef __int16 CANDIDATES;
typedef __int8* BOARDS;

#endif