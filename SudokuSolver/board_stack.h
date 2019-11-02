#include "sudoku.h"

#ifndef BOARD_STACK_H_
#define BOARD_STACK_H_

typedef struct stack_node_t {
	board_t board;
	stack_node_t* next;
} stack_node_t;

typedef stack_node_t* stack_t;

int stack_empty(stack_t node);
void stack_push(stack_t* stack, board_t board);
void stack_pop(stack_t* stack, board_t* board);
void stack_free(stack_t* stack);

#endif