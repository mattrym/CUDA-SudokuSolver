#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "sudoku.h"
#include "gpu_solver.h"

inline void check_cuda_error(cudaError_t cuda_status, const char* file, int line)
{
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "CUDA error (%s:%d): %s", file, line, cudaGetErrorString(cuda_status));
		exit(EXIT_FAILURE);
	}
}
#define CUDA_SAFE(cuda_status) check_cuda_error(cuda_status, __FILE__, __LINE__)

__device__ int cell_index(const int cell)
{
	int row, col;
	int mask_type, mask_offset;

	mask_type = threadIdx.x / N;
	mask_offset = threadIdx.x % N;

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

__device__ void calculate_masks(const BOARD board, MASK* masks, BLOCK_FLAGS* flags)
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
	MASK row_mask, col_mask, sub_mask;

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

__device__ void fork_board(BOARD boards, int* block_flags, BOARD board, CANDIDATES candidates, int fork_cell)
{
	int digit, forks, block;
	int block_id;
	BOARD forked_board;

	forks = 0;
	block = 0;

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
						forked_board = boards + block * BOARD_SIZE;

						memcpy(forked_board, board, BOARD_SIZE);
						forked_board[fork_cell] = digit + 1;
						block_flags[block] = BUSY;

						break;
					}
				}

			}
			else
			{
				forked_board = boards + blockIdx.x * BOARD_SIZE;

				memcpy(forked_board, board, BOARD_SIZE);
				forked_board[fork_cell] = digit + 1;
			}
		}
	}
}

__global__ void solve_kernel(BOARDS boards, BLOCK_STATUS* block_status)
{
	__shared__ MASK masks[N * n];
	__shared__ CANDIDATES candidates[BOARD_SIZE];
	__shared__ BLOCK_FLAGS flags;

	__shared__ int min_forks;
	__shared__ int fork_cell;

	__shared__ CELL board[BOARD_SIZE];
	BOARD last_board;

	if (block_status[blockIdx.x] != BUSY)
	{
		return;
	}

	if (!threadIdx.x)
	{
		min_forks = 9;
		fork_cell = BOARD_SIZE;

		memcpy(board, boards + blockIdx.x * BOARD_SIZE, BOARD_SIZE);
		last_board = boards + BLOCKS * BOARD_SIZE;
	}
	__syncthreads();

	do
	{
		flags = { 0, 1, 0 };

		calculate_masks(board, masks, &flags);
		__syncthreads();

		find_candidates(board, masks, candidates, &flags);
		__syncthreads();
	} while (!flags.error && !flags.success && flags.progress);

	if (flags.error)
	{
		if (!threadIdx.x)
		{
			block_status[blockIdx.x] = IDLE;
		}
		return;
	}

	if (flags.success)
	{
		if (!threadIdx.x)
		{
			block_status[BLOCKS] = SUCCESS;
			memcpy(last_board, board, BOARD_SIZE * sizeof(CELL));
		}
		return;
	}

	find_fork_cell(board, candidates, &min_forks, &fork_cell);
	__syncthreads();

	if (!threadIdx.x)
	{
		fork_board(boards, block_status, board, candidates[fork_cell], fork_cell);
	}
}

void solve_gpu(const BOARD input_board, BOARD output_board)
{
	BOARDS boards;
	BLOCK_STATUS* block_status;
	BLOCK_STATUS last_block_status;

	int it;

	CUDA_SAFE(cudaSetDevice(0));

	CUDA_SAFE(cudaMalloc((void**)&block_status, (BLOCKS + 1) * sizeof(BLOCK_STATUS)));
	CUDA_SAFE(cudaMemset(block_status, 1, 1));

	CUDA_SAFE(cudaMalloc((void**)&boards, (BLOCKS + 1) * BOARD_SIZE * sizeof(CELL)));
	CUDA_SAFE(cudaMemcpy(boards, input_board, BOARD_SIZE * sizeof(CELL), cudaMemcpyHostToDevice));

	for (it = 0; it < ITERATIONS; ++it)
	{
		solve_kernel<<<BLOCKS, THREADS>>>(boards, block_status);
		CUDA_SAFE(cudaGetLastError());
		CUDA_SAFE(cudaDeviceSynchronize());

		CUDA_SAFE(cudaMemcpy(&last_block_status, block_status + BLOCKS, sizeof(BLOCK_STATUS), cudaMemcpyDeviceToHost));
		if (last_block_status == SUCCESS)
		{
			CUDA_SAFE(cudaMemcpy(output_board, boards + BLOCKS * BOARD_SIZE, BOARD_SIZE * sizeof(CELL), cudaMemcpyDeviceToHost));
			break;
		}
	}

	cudaFree(boards);
	cudaFree(block_status);

	CUDA_SAFE(cudaDeviceReset());
}