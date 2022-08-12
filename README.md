# phase_retrieval_algorithms

Pytorch implementation of phase retrieval algorithms.
Currently, Error reduction & Hybrid input-output algorithms are implemented.

## Requirements
- pytorch
- torchvision
- numpy
- PIL
- tqdm

## Usage

### Required argument

```
--algorithm         ER or HIO

--num_iterations    Number of iterations for each algorithm.

--num_repeats       Number of repeats for each sample.

--data_root         Data directory

--batch_size        Size of mini-batch

--gpu               Index for gpu device.
```