#pragma once
#include <CL/opencl.h>
#include <random_generator/precalc.h>

void matvec_inplace(unsigned int *vector, unsigned int *matrix)
{
  const int N = 5;
  unsigned int result[N] = { 0 };
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < 32; j++) {
      if(vector[i] & (1 << j)) {
        for(int k = 0; k < N; k++) {
          result[k] ^= matrix[N * (i * 32 + j) + k];
        }
      }
    }
  }
  for(int i = 0; i < N; i++) {
    vector[i] = result[i];
  }
}

typedef struct t_xorwow_state {
  cl_uint x[5];
  cl_uint d;
} xorwow_state;

void skipahead_sequence(unsigned int seq, xorwow_state *state) {
  int matrix_num = 0;
  while(seq) {
    for(unsigned int t = 0; t < (seq & PRECALC_BLOCK_MASK); t++) {
      matvec_inplace(state->x, precalc_xorwow_matrix[matrix_num]);
    }
    seq >>= PRECALC_BLOCK_SIZE;
    matrix_num++;
  }
}

void init_xorwow(xorwow_state *state, unsigned long long seed, unsigned long long sequence) {
  unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
  unsigned int s1 = (unsigned int)(sequence >> 32) ^ 0xf7dcefddUL;
  unsigned int t0 = 1099087573UL * s0;
  unsigned int t1 = 2591861531UL * s1;
  state->d = 6615241 + t1 + t0;
  state->x[0] = 123456789UL + t0;
  state->x[1] = 362436069UL ^ t0;
  state->x[2] = 521288629UL + t1;
  state->x[3] = 88675123UL ^ t1;
  state->x[4] = 5783321UL + t0;
  skipahead_sequence(sequence, state);
}

unsigned int rand(xorwow_state *state) {
  unsigned int t;
  t = (state->x[0] ^ (state->x[0] >> 2));
  state->x[0] = state->x[1];
  state->x[1] = state->x[2];
  state->x[2] = state->x[3];
  state->x[3] = state->x[4];
  state->x[4] = (state->x[4] ^ (state->x[4] <<4)) ^ (t ^ (t << 1));
  state->d += 362437;
  return state->x[4] + state->d;
}
