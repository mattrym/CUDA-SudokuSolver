#include <stdlib.h>
#include <string.h>

#include "board_stack.h"
#include "sudoku.h"

#include "cpu_solver.h"

board_t clone_board(board_t board)
{
	board_t new_board;

	new_board = (board_t)calloc(BOARD_SIZE, sizeof(cell_t));
	if (!new_board)
	{
		perror("Error while allocating memory for stacked board");
		exit(EXIT_FAILURE);
	}

	return (board_t)memcpy(new_board, board, BOARD_SIZE);
}

int get_cell_index(int mask_index, const int cell)
{
	int row, col;
	int mask_type, mask_offset;

	mask_type = mask_index / N;
	mask_offset = mask_index % N;

	switch (mask_type)
	{
	case ROW_MASK:
		row = mask_offset;
		col = cell;
		break;
	case COL_MASK:
		row = cell;
		col = mask_offset;
		break;
	case SUB_MASK:
		row = (mask_offset / n) * n + cell / n;
		col = (mask_offset % n) * n + cell % n;
		break;
	}

	return row * N + col;
}

void calculate_masks(const board_t board, mask_t* masks, flags_t* flags)
{
	int cell, cell_index;
	int value, value_mask;
	int mask_index;

	for (mask_index = 0; mask_index < N * n; ++mask_index)
	{
		masks[mask_index] = 0;

		for (cell = 0; cell < N; ++cell)
		{
			cell_index = get_cell_index(mask_index, cell);
			if (value = board[cell_index])
			{
				if ((value_mask = 1 << (value - 1)) & masks[mask_index])
				{
					flags->error = 1;
					return;
				}
				masks[mask_index] |= value_mask;
			}
		}
	}
}

void find_candidates(board_t board, mask_t* masks, candidates_t* candidates, flags_t* flags)
{
	int row, col, sub;
	int cell, digit;
	int progress, success;

	mask_t row_mask, col_mask, sub_mask;

	progress = 0;
	success = 1;

	for (cell = 0; cell < BOARD_SIZE; ++cell)
	{
		row = cell / N;
		col = cell % N;
		sub = (row / n) * n + (col / n);

		row_mask = masks[ROW_MASK * N + row];
		col_mask = masks[COL_MASK * N + col];
		sub_mask = masks[SUB_MASK * N + sub];

		candidates[cell] = row_mask | col_mask | sub_mask;

		if (!board[cell])
		{
			for (digit = 0; digit < N; ++digit)
			{
				if (!(candidates[cell] ^ (1 << digit) ^ FULL_MASK))
				{
					board[cell] = (digit + 1);
					flags->progress = 1;
					break;
				}
			}
			if (!board[cell])
			{
				flags->success = 0;
			}
		}
	}
}

int find_fork_cell(board_t board, candidates_t* candidates)
{
	int digit, cell, fork_cell;
	int forks, min_forks;

	min_forks = N + 1;
	fork_cell = BOARD_SIZE;

	for (cell = 0; cell < BOARD_SIZE; ++cell)
	{
		forks = N;
		if (!board[cell])
		{
			for (digit = 0; digit < N; ++digit)
			{
				forks -= ((candidates[cell] >> digit) & 1);
			}
		}

		if (forks < min_forks)
		{
			min_forks = forks;
			fork_cell = cell;
		}
	}

	return fork_cell;
}

void fork_board(stack_t* stack, board_t board, candidates_t candidates, int fork_cell)
{
	int digit;
	board_t forked_board;

	for (digit = N - 1; digit >= 0; --digit)
	{
		if (!((candidates >> digit) & 1))
		{
			forked_board = clone_board(board);
			forked_board[fork_cell] = digit + 1;
			stack_push(stack, forked_board);
		}
	}
}

void solve_cpu(board_t input_board, board_t output_board)
{
	stack_t stack;
	flags_t flags;
	board_t curr_board;

	mask_t masks[N * n];
	candidates_t candidates[BOARD_SIZE];

	int fork_cell;

	stack = NULL;
	curr_board = clone_board(input_board);
	stack_push(&stack, curr_board);

	while (!stack_empty(stack))
	{
		stack_pop(&stack, &curr_board);

		do {
			flags = { 0, 1, 0 };
			calculate_masks(curr_board, masks, &flags);
			find_candidates(curr_board, masks, candidates, &flags);
		} while (flags.progress && !flags.success && !flags.error);

		if (!flags.error && flags.success)
		{
			break;
		}

		if (!flags.error && !flags.progress)
		{
			fork_cell = find_fork_cell(curr_board, candidates);
			fork_board(&stack, curr_board, candidates[fork_cell], fork_cell);
		}

		free(curr_board);
	}

	if (flags.success)
	{
		memcpy(output_board, curr_board, BOARD_SIZE);
		free(curr_board);
	}

	stack_free(&stack);
}
