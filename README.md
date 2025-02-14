# matmul-gpu
**A fast and efficient WebGPU powered implementation of matrix multiplication.**

`matmul-gpu` provides a high-performance implementation of matrix multiplication using WebGPU.  WebGPU is a modern web API that allows for GPU-accelerated computations directly within the browser, leading to significant performance gains for operations like matrix multiplication compared to traditional CPU-based JavaScript.

**NOTE:** This package is experimental. Please use with caution, and report any bugs you find. 

[![npm version](https://badge.fury.io/js/matmul-gpu.svg)](https://badge.fury.io/js/matmul-gpu)

## Features

- **Fast Multiplication of Large Matrices:**  Quickly multiplies large matrices in the browser.
- **Memory Efficient Batching:** Supports batch processing for improved memory efficiency when dealing with extremely large matrices.
- **CPU Fallback:**  Automatically falls back to a CPU-based implementation if WebGPU is not available, ensuring compatibility.

## Import

Install using npm:

```
npm install mathjs
```

Or import from a CDN (e.g. jsDelivr):

```js
import { matrixMultiply } from 'https://cdn.jsdelivr.net/npm/matmul-gpu/+esm'
```

## Usage 

``matmul-gpu`` exports the primary function: ``matrixMultiply``. This function takes two 2D arrays (representing matrices) as input and returns a new 2D array representing their matrix product. The function is asynchronous due to its reliance on the asynchronous WebGPU API.  

```js
// Result will be [[19,22],[43,50]]
const product = await matrixMultiply([[1,2],[3,4]], [[5,6],[7,8]]);
```


For very large matrices, you can provide an optional argument: the ``batchSize``.  This parameter controls the size of the batches used during the computation. Batching prevents memory errors that can occur when loading excessively large matrices into WebGPU. The default ``batchSize`` is 1024 (which corresponds to a [1024, 1024] batch shape). You can also set a 2D batch shape with a two-element array; for example, [256, 64] would create batches of 256 rows and 64 columns

```js
const product1 = await matrixMultiply(largeMatrixA, largeMatrixB, { batchSize: 256 } );
const product2 = await matrixMultiply(largeMatrixA, largeMatrixB, { batchSize: [256, 64] } );
```

Using a GPU introduces a small overhead. Consequently, CPU computation is faster for multiplying smaller matrices. You can define a threshold, expressed as the number of operations (estimated as N x M x K for N x M and M x K matrices), above which the function will utilize the GPU. The default threshold is 373,248 (equivalent to 72 x 72 x 72).

```js
const product = await matrixMultiply(largeMatrixA, largeMatrixB, { operationsGpuThreshold: 373248 } );
```

``matmul-gpu`` also exports the helper function ``isGpuAvailable`` which returns ``true`` if a GPU is available and ``false`` otherwise. This function is also asynchronous. 

```js
const gpuAvailable = await isGpuAvailable();
```
