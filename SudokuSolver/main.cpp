#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#include "board_io.h"
#include "sudoku.h"
#include "cpu_solver.h"
#include "gpu_solver.h"

void measure_time(void(solve_fun)(board_t, board_t), board_t input_board, board_t output_board)
{
	LARGE_INTEGER freq, start, end;
	double elapsed_time;

	QueryPerformanceFrequency(&freq);

	QueryPerformanceCounter(&start);
	solve_fun(input_board, output_board);
	QueryPerformanceCounter(&end);

	elapsed_time = (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

	print_board(stdout, output_board);
	printf("\nTime: %.7g ms\n", elapsed_time);
}

int main(int argc, char* argv[])
{
	char* filename;
	board_t input_board, cpu_output_board, gpu_output_board;

	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s FILENAME\n", argv[0]);
		exit(1);
	}
	filename = argv[1];

	input_board = (board_t)calloc(BOARD_SIZE, sizeof(cell_t));
	if (!input_board)
	{
		perror("Error while allocating memory for input board");
		exit(1);
	}

	cpu_output_board = (board_t)calloc(BOARD_SIZE, sizeof(cell_t));
	if (!cpu_output_board)
	{
		perror("Error while allocating memory for GPU output board");
		exit(1);
	}

	gpu_output_board = (board_t)calloc(BOARD_SIZE, sizeof(cell_t));
	if (!gpu_output_board)
	{
		perror("Error while allocating memory for GPU output board");
		exit(1);
	}

	if (load_board(filename, input_board))
	{
		fprintf(stderr, "Error while loading board: invalid input format\n");
		exit(1);
	}

	printf("------INPUT------\n");
	print_board(stdout, input_board);
	printf("-----------------\n\n");

	printf("-------CPU-------\n");
	measure_time(solve_cpu, input_board, cpu_output_board);
	printf("-----------------\n\n");

	printf("-------GPU-------\n");
	measure_time(solve_gpu, input_board, gpu_output_board);
	printf("-----------------\n\n");

	free(input_board);
	free(cpu_output_board);
	free(gpu_output_board);

	return 0;
}