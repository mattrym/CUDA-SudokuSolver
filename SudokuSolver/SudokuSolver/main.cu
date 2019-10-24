#include <stdio.h>
#include <stdlib.h>

#include "sudoku.cuh"
#include "solver.cuh"
#include "board_io.cuh"

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
	print_board(input_board);

	iterations = 0;
	run_solve(input_board, output_board, &iterations);

	printf("Output board:\n");
	print_board(output_board);

	return 0;
}