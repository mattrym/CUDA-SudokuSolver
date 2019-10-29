#include "sudoku.h"

#ifndef BOARD_STACK_H_
#define BOARD_STACK_H_

typedef struct STACK_NODE {
	BOARD board;
	STACK_NODE* next;
} STACK_NODE;

typedef STACK_NODE* STACK;

int stack_empty(STACK node);
void stack_push(STACK* stack, BOARD board);
void stack_pop(STACK* stack, BOARD* board);
void stack_free(STACK* stack);

#endif