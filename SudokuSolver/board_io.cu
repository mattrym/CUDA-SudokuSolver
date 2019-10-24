#include <stdio.h>
#include <stdlib.h>

#include "sudoku.cuh"
#include "board_io.cuh"

int read_char(FILE*, char*, int, int);

int load_board(char* filename, BOARD board)
{
	FILE* file;
	int row = 0, col = 0;

	if ((file = fopen(filename, "r")) == NULL)
	{
		perror("Error while opening a file");
		exit(1);
	}

	while (row < N)
	{
		if (read_char(file, &(board[row * N + col]), row, col))
		{
			return 1;
		}

		row += (++col / N);
		col %= N;
	}

	if (ferror(file))
	{
		perror("Error while reading a file");
		exit(1);
	}

	fclose(file);
	return 0;
}

int read_char(FILE* file, char* mem, int row, int col)
{
	char buf[2];
	char* format;

	format = (col == N - 1) ? (row == N - 1) ? "%c" : "%c\n" : "%c ";
	if (fscanf(file, format, buf) == 0)
	{
		if (feof(file))
		{
			return 1;
		}
		perror("Error while reading file");
		exit(1);
	}

	if (buf[0] >= '0' && buf[0] <= '9')
	{
		*mem = buf[0] - '0';
		return 0;
	}
	return 1;
}

void print_board(BOARD board, FILE* fd)
{
	int row, col;
	int value;

	for (row = 0; row < N; ++row)
	{
		for (col = 0; col < N - 1; ++col)
		{
			value = board[row * N + col];
			fprintf(fd, "%d ", value);
		}

		value = board[row * N + col];
		fprintf(fd, "%d\n", value);
	}
}