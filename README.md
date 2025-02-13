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

matmul-gpu currently exports a single function: ``matrixMultiply``. This function takes two 2D arrays (representing matrices) as input and returns a new 2D array representing their matrix product.

```js
// Result will be [[19,22],[43,50]]
const product = await matrixMultiply([[1,2],[3,4]], [[5,6],[7,8]]);
```

The ``matrixMultiply`` function is asynchronous due to its reliance on the asynchronous WebGPU API.  You must use await when calling it.

For very large matrices, you can provide an optional third argument: the ``batchSize``.  This parameter controls the size of the batches used during the computation. Batching prevents memory errors that can occur when loading excessively large matrices into WebGPU. The default ``batchSize`` is 1024.

```js
const product = await matrixMultiply(largeMatrixA, largeMatrixB, 256);
```
