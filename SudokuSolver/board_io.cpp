#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "sudoku.h"
#include "board_io.h"

int read_char(FILE* file, char* mem, int row, int col)
{
	char buf[2];
	char* format;

	format = (col == N - 1) ? (row == N - 1) ? "%c" : "%c\n" : "%c ";
	if (fscanf_s(file, format, buf, _countof(buf)) == 0)
	{
		if (feof(file))
		{
			return 1;
		}

		perror("Error while reading file");
		fclose(file);
		exit(1);
	}

	if (buf[0] >= '0' && buf[0] <= '9')
	{
		*mem = buf[0] - '0';
		return 0;
	}
	return 1;
}

int load_board(char* filename, BOARD board)
{
	FILE* file;
	int row = 0, col = 0;

	if (fopen_s(&file, filename, "r") != 0)
	{
		fprintf(stderr, "Error while opening a file: %s\n", filename);
		exit(1);
	}

	while (row < N)
	{
		if (read_char(file, &(board[row * N + col]), row, col))
		{
			fclose(file);
			return 1;
		}

		row += (++col / N);
		col %= N;
	}

	if (ferror(file))
	{
		perror("Error while reading a file");
		fclose(file);
		exit(1);
	}

	fclose(file);
	return 0;
}

void print_board(FILE* fd, BOARD board)
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