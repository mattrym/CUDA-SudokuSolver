#include <stdio.h>
#include <stdlib.h>

#include "board_io.h"
#include "sudoku.h"
#include "gpu_solver.h"

int main(int argc, char* argv[])
{
	int iterations;
	char* filename;
	BOARD input_board, output_board;

	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s FILENAME\n", argv[0]);
		exit(1);
	}
	filename = argv[1];

	input_board = (BOARD)calloc(N * N, sizeof(__int8));
	if (!input_board)
	{
		perror("Error while allocating memory for input board");
		exit(1);
	}

	output_board = (BOARD)calloc(N * N, sizeof(__int8));
	if (!output_board)
	{
		perror("Error while allocating memory for output board");
		exit(1);
	}

	if (load_board(filename, input_board))
	{
		fprintf(stderr, "Error while loading board: invalid input format\n");
		exit(1);
	}

	printf("Input board:\n");
	print_board(stdout, input_board);

	iterations = 0;
	run_solve(input_board, output_board);

	printf("Output board:\n");
	print_board(stdout, output_board);

	return 0;
}