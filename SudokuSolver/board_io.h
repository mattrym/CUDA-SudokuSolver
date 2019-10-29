#ifndef BOARD_IO_H_
#define BOARD_IO_H_

#include <stdio.h>
#include "sudoku.h"

int load_board(char* filename, BOARD board);
void print_board(FILE* file, BOARD board);

#endif