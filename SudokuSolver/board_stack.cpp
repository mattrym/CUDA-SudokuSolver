#include <stdlib.h>

#include "board_stack.h"

int stack_empty(STACK node)
{
	return !node;
}

void stack_push(STACK* stack, BOARD board)
{
	STACK_NODE* node;
	node = (STACK_NODE*) malloc(sizeof(STACK_NODE));

	*node = { board, *stack };
	*stack = node;
}

void stack_pop(STACK* stack, BOARD* board)
{
	STACK_NODE* node;
	node = *stack;

	if (!stack_empty(node))
	{
		*board = node->board;
		*stack = node->next;
		free(node);
	}
}

void stack_free(STACK* stack)
{
	STACK_NODE* node;
	
	while (!stack_empty(node = *stack))
	{
		*stack = node->next;
		free(node->board);
		free(node);
	}
}