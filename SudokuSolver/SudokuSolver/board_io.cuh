#include <stdio.h>

#ifndef BOARD_IO_CUH_
#define BOARD_IO_CUH_

int load_board(char* filename, BOARD board);
void print_board(BOARD board, FILE* file = stdout);

#endif