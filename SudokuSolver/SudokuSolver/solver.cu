#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include "sudoku.cuh"
#include "solver.cuh"


__device__ int cell_index(int cell)
{
	int row, col;
	int mask_type, mask_index;

	mask_type = threadIdx.x / N;
	mask_index = threadIdx.x % N;

	switch (mask_type)
	{
	case ROW_MASK:
		row = mask_index;
		col = cell;
		break;
	case COL_MASK:
		row = cell;
		col = mask_index;
		break;
	case SUB_MASK:
		row = (mask_index / n) * n + cell / n;
		col = (mask_index % n) * n + cell % n;
		break;
	}

	return row * N + col;
}

__device__ void calculate_masks(BOARD board, MASK* masks, BLOCK_FLAGS* flags)
{
	int cell, cell_i, value, value_mask;

	masks[threadIdx.x] = 0;

	for (cell = 0; cell < N; ++cell)
	{
		cell_i = cell_index(cell);
		if (value = board[cell_i])
		{
			if ((value_mask = 1 << (value - 1)) & masks[threadIdx.x])
			{
				flags->error = 1;
				break;
			}
			masks[threadIdx.x] |= value_mask;
		}
	}
}

__device__ void find_candidates(BOARD board, MASK* masks, CANDIDATES* candidates, BLOCK_FLAGS* flags)
{
	int row, sub;
	int col, lcol, ucol;
	int cell, digit;
	__int16 row_mask, col_mask, sub_mask;

	row = threadIdx.x / n;
	sub = (threadIdx.x / N) * n + threadIdx.x % n;

	row_mask = masks[ROW_MASK * N + row];
	sub_mask = masks[SUB_MASK * N + sub];

	lcol = (threadIdx.x % n) * n;
	ucol = lcol + n;

	for (col = lcol; col < ucol; ++col)
	{
		col_mask = masks[COL_MASK * N + col];
		cell = row * N + col;
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

__device__ void find_fork_cell(BOARD board, CANDIDATES* candidates, int* min_forks, int* fork_cell)
{
	int digit, cell, offset;
	int forks[n];

	offset = threadIdx.x * n;
	for (cell = 0; cell < n; ++cell)
	{
		forks[cell] = N;
		if (!board[offset + cell])
		{
			for (digit = 0; digit < N; ++digit)
			{
				forks[cell] -= ((candidates[offset + cell] >> digit) & 1);
			}
		}

		__iAtomicMin(min_forks, forks[cell]);
	}
	__syncthreads();

	for (cell = 0; cell < n; ++cell)
	{
		if (forks[cell] == *min_forks)
		{
			__iAtomicMin(fork_cell, offset + cell);
		}
	}
}

__device__ void fork_board(BOARD boards, int* block_flags, CANDIDATES candidates, int fork_cell)
{
	int digit, forks, block;
	int block_id;
	BOARD board, forked_board;

	forks = 0;
	block = 0;

	board = boards + blockIdx.x * BOARD_SIZE;

	for (digit = 0; digit < N; ++digit)
	{
		if (!((candidates >> digit) & 1))
		{
			if (forks++)
			{
				for (; block < gridDim.x; ++block)
				{
					block_id = gridDim.x * blockIdx.x + fork_cell + SUCCESS;
					__iAtomicCAS(&block_flags[block], IDLE, block_id);

					if (block_flags[block] == block_id)
					{
						forked_board = boards + block * BOARD_SIZE * sizeof(CELL);

						memcpy(forked_board, board, BOARD_SIZE * sizeof(CELL));
						forked_board[fork_cell] = digit + 1;
						block_flags[block] = BUSY;

						break;
					}
				}

			}
			else
			{
				board[fork_cell] = digit + 1;
			}
		}
	}
}

__global__ void solve(BOARDS boards, BLOCK_STATUS* block_status)
{
	__shared__ MASK masks[N * n];
	__shared__ CANDIDATES candidates[BOARD_SIZE];
	__shared__ BLOCK_FLAGS flags;

	__shared__ int min_forks;
	__shared__ int fork_cell;

	BOARD board, last_board;

	board = boards + blockIdx.x * BOARD_SIZE;
	last_board = boards + BLOCKS * BOARD_SIZE;
	
	min_forks = 9;
	fork_cell = BOARD_SIZE;

	if (block_status[blockIdx.x] != BUSY)
	{
		return;
	}

	do
	{
		flags = { 0, 1, 0 };

		calculate_masks(board, masks, &flags);
		__syncthreads();

		find_candidates(board, masks, candidates, &flags);
		__syncthreads();
	} while (!flags.error && !flags.success && flags.progress);

	if (flags.success)
	{
		if (!threadIdx.x)
		{
			block_status[BLOCKS] = SUCCESS;
			memcpy(last_board, board, BOARD_SIZE * sizeof(CELL));
		}
		return;
	}

	if (flags.error)
	{
		if (!threadIdx.x)
		{
			block_status[blockIdx.x] = IDLE;
		}
		return;
	}

	find_fork_cell(board, candidates, &min_forks, &fork_cell);
	__syncthreads();

	if (!threadIdx.x)
	{
		fork_board(boards, block_status, candidates[fork_cell], fork_cell);
	}
}

cudaError_t run_solve(BOARD input_board, BOARD output_board, int* iterations1)
{
	BOARDS boards;
	BLOCK_STATUS* block_status;

	cudaError_t cuda_status;
	int iterations, result;

	cuda_status = cudaMalloc((void**)&block_status, (BLOCKS + 1) * sizeof(BLOCK_STATUS));
	if (cuda_status != cudaSuccess) {
		fprintf(stdout, "cudaMalloc failed!");
		goto Error;
	}

	cuda_status = cudaMalloc((void**)&boards, (BLOCKS + 1) * BOARD_SIZE * sizeof(CELL));
	if (cuda_status != cudaSuccess) {
		fprintf(stdout, "cudaMalloc failed!");
		goto Error;
	}

	cuda_status = cudaMemset(block_status, 1, 1);
	if (cuda_status != cudaSuccess) {
		fprintf(stdout, "cudaMemset failed!");
		goto Error;
	}

	cuda_status = cudaMemcpy(boards, input_board, BOARD_SIZE * sizeof(CELL), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stdout, "cudaMemcpy failed!");
		goto Error;
	}

	for (iterations = 0; iterations < ITERATIONS; iterations++)
	{
		solve <<<BLOCKS, THREADS>>>(boards, block_status);

		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stdout, "solve launch failed: %s\n", cudaGetErrorString(cuda_status));
			goto Error;
		}

		cuda_status = cudaDeviceSynchronize();
		if (cuda_status != cudaSuccess) {
			fprintf(stdout, "cudaDeviceSynchronize returned error code %d after launching solve!\n", cuda_status);
			goto Error;
		}

		cuda_status = cudaMemcpy(&result, block_status + BLOCKS, sizeof(BLOCK_STATUS), cudaMemcpyDeviceToHost);
		if (cuda_status != cudaSuccess) {
			fprintf(stdout, "cudaMemcpy failed!");
			goto Error;
		}

		if (result == SUCCESS)
		{
			cuda_status = cudaMemcpy(output_board, boards + BLOCKS * BOARD_SIZE, BOARD_SIZE * sizeof(CELL), cudaMemcpyDeviceToHost);
			if (cuda_status != cudaSuccess) {
				fprintf(stdout, "cudaMemcpy failed!");
				goto Error;
			}

			break;
		}
	}

Error:
	cudaFree(boards);
	cudaFree(block_status);
	return cuda_status;
}