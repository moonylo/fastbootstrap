#define RAND_2POW32_INV (2.3283064e-10f)
#define RAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)

typedef struct t_xorwow_state {
  unsigned int x[5];
  unsigned int d;
} xorwow_state;

unsigned int rand_kernel(xorwow_state *state) {
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

float _rand_uniform(unsigned int x) {
  return x * RAND_2POW32_INV + (RAND_2POW32_INV/2.0f);
}

float rand_uniform(xorwow_state * state)
{
  return _rand_uniform(rand_kernel(state));

}

double _rand_uniform_double_hq(unsigned int x, unsigned int y)
{
    unsigned long long z = (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
    return z * RAND_2POW53_INV_DOUBLE + (RAND_2POW53_INV_DOUBLE/2.0);
}

double rand_uniform_double(xorwow_state * state)
{
  unsigned int x, y;
  x = rand_kernel(state);
  y = rand_kernel(state);
  return _rand_uniform_double_hq(x, y);
}


__kernel void bootstrap_kernel(__global xorwow_state* rand_states, const int bootstrap_samples, __global float *output, __global float *values, const int nr_of_values) {
    int i = get_global_id(0);
    float sum = 0;

    if(i < bootstrap_samples) {
      xorwow_state local_xorwow_state = rand_states[i];
      for(int j = 0; j < nr_of_values; j++) {
        sum += values[(int) floor(rand_uniform(&local_xorwow_state) * nr_of_values + 0.999999 - 1)];
      }
      output[i] = sum / nr_of_values;
    }

}

__kernel void gen_random_kernel_int(__global xorwow_state* rand_states, __global int *output, const int n) {
    int i = get_global_id(0);

    if(i < n) {
      xorwow_state local_xorwow_state = rand_states[i];
      output[i] = rand_kernel(&local_xorwow_state);
    }

}

__kernel void gen_random_kernel_float(__global xorwow_state* rand_states, __global float *output, const int n) {
    int i = get_global_id(0);

    if(i < n) {
      xorwow_state local_xorwow_state = rand_states[i];
      output[i] = rand_uniform(&local_xorwow_state);
    }

}

__kernel void gen_random_kernel_double(__global xorwow_state* rand_states, __global double *output, const int n) {
    int i = get_global_id(0);

    if(i < n) {
      xorwow_state local_xorwow_state = rand_states[i];
      output[i] = rand_uniform_double(&local_xorwow_state);
    }

}

