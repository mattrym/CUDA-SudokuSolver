#include <stdlib.h>

#include "board_stack.h"

int stack_empty(stack_t node)
{
	return !node;
}

void stack_push(stack_t* stack, board_t board)
{
	stack_node_t* node;
	node = (stack_node_t*)malloc(sizeof(stack_node_t));

	*node = { board, *stack };
	*stack = node;
}

void stack_pop(stack_t* stack, board_t* board)
{
	stack_node_t* node;
	node = *stack;

	if (!stack_empty(node))
	{
		*board = node->board;
		*stack = node->next;
		free(node);
	}
}

void stack_free(stack_t* stack)
{
	stack_node_t* node;
	
	while (!stack_empty(node = *stack))
	{
		*stack = node->next;
		free(node->board);
		free(node);
	}
}