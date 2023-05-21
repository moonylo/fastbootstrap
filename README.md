# fastbootstrap

This package uses the GPU via OpenCL to calculate ordinary, non-parametric bootstraped means of a given vector.

# Installation & System Requirements

## Windows
* You need an OpenCL 1.2 capable device. The code is currently hardcoded for GPUs.
* You need an OpenCL runtime installed 
    * For Nvidia, this comes with the GPU driver.
* If you want to build the package from source, you need the OpenCL.dll. For all my computers this has already been in `C:\Windows\System32\OpenCL.dll`. If this is not the case for you, let me know / let me know how you got it installed.

## Unix
* Not attempted yet. But if it works on Windows so easily, Unix should be no problem.


# Random Number Generator

The code runs on the GPU and thus can't access R's random functions. So a pseudo-random generator had to be implemented.
* CUDAs XORWOW implementation in cuRand is used.
    * For every bootstrap replication a different sequence is created, which allows for non-overlapping periods of 2^67-1.
    * Theoretically that would leave space for 2^123 distinct sequences, but due to the implementation it already breaks around 4-5 billion from my testing.
* The same seed will produce the same random numbers every time.
    * This allows you to use this tool for paired observations.
    * Or for metrics, which need the mean / sum of multiple variables (e.g. if your metric is `mean(x) / mean(y)`).

# Usage

```r
require(fastbootstrap)

df <- data.frame(x1 = rnorm(5000, 50))
replications <- 10000L
seed <- 2023L

# Create an instance of the bootstrap manager and initialize it with the number of replications and the seed
bs_mgr <- new(opencl_bootstrap_manager_float, replications, seed)
# Run bootstrap for mean(df$x1)
output <- bs_mgr$get_bootstrapped_means(df$x1)

# Compare output with the simplest version in base R
bs_classic <- replicate(replications, mean(sample(df$x1, replace = TRUE)))
plot(density(output), col = "red")
lines(density(bs_classic))

# Change the number of bootstrap samples and/or the seed
replications <- 20000L
seed <- 0L
bs_mgr$set_parameters(replications, seed)
```

# Performance

Tested on a Nvidia GTX 3080.

```r
library(microbenchmark)
library(boot)
library(bootstrap)
library(fastbootstrap)

set.seed(0)
df <- data.frame(x1 = rnorm(5000, 50))

replications <- 10000L
bs_mgr <- new(opencl_bootstrap_manager_float, replications, 2023L)

means <- function(x, i) {
  mean(x)
}

microbenchmark(
  boot(data = df$x1, statistic = means, R=replications, parallel = "multicore")
  , replicate(replications, mean(sample(df$x1, replace=TRUE)))
  , bs_mgr$get_bootstrapped_means(df$x1)
  , times = 20
)

#                                                                             expr         min          lq      mean      median        uq       max neval
# boot(data = df$x1, statistic = means, R = replications, parallel = "multicore") 2648.960900 2713.500551 2712.6838 2721.989501 2726.9062 2765.6686    20
#                    replicate(replications, mean(sample(df$x1, replace = TRUE))) 2851.856701 2868.868501 2893.8785 2893.932051 2910.8763 2974.1622    20
#                                            bs_mgr$get_bootstrapped_means(df$x1)    1.376801    5.968501   60.5336    9.865701  133.3273  274.9604    20
```

# License

MIT + file LICENSE
