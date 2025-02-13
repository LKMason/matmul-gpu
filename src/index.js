
let gpuDevice = null;

async function getGpuDevice() {
  try {
    if (!( "gpu" in navigator)) {
      console.warn("WebGPU not supported. CPU computation will be performed instead.")
    }
    const adapter = await navigator.gpu.requestAdapter();
    gpuDevice = await adapter.requestDevice();
    return gpuDevice;
  } catch (e) {
    console.error(e);
    console.warn("A problem occured setting up interface with GPU device. CPU computation will be performed instead.");
    return null;
  }
}

export async function isGpuAvailable() {
  return await getGpuDevice() != null;
}

/**
 * Multiplies two 2D matrices, optionally using batched processing.
 * Uses WebGPU if available; otherwise falls back on a JS implementation.
 *
 * @async
 * @function matrixMultiply2D
 * @param {number[][]} matrix1 The first matrix (N x M).
 * @param {number[][]} matrix2 The second matrix (M x K).
 * @param {object} [options={}] An object containing optional parameters.
 * @param {number | number[]} [options.batchSize=1024] The size of the batches to use for processing.  If a single number is provided, it's used as both the row and column batch size. If an array of two numbers is provided, the first number is the row batch size and second is the column batch size. If null, uses the maximum dimension of the matrices, effectively no batching.  Must be greater than 1 if specified.
 * @param {number} [options.operationsGpuThreshold=373248] Number of operations (N x M x K) above which the GPU is used.
 * @returns {number[][]} The resulting matrix (N x K).
 * @throws {Error} If the inner dimensions of the matrices do not match.
 * @throws {Error} If the batch size is less than 2.
 */
export async function matrixMultiply(matrix1, matrix2, options={}) {
  let { 
    batchSize = 1024, 
    operationsGpuThreshold = 373248, // 72 x 72 x 72
  } = options;

  const N = matrix1.length;
  const M = matrix1[0].length;
  const M2 = matrix2.length;
  const K = matrix2[0].length;

  if (M !== M2) {
    throw new Error("Inner dimensions of matrices must match for multiplication");
  }

  if (batchSize == null) {
    batchSize = [Math.max(N, M, K), Math.max(N, M, K)];
  }

  if (!Array.isArray(batchSize)) {
    batchSize = [batchSize, batchSize];
  }

  if (batchSize[0] < 2 || batchSize[1] < 2) {
    throw new Error("Batch size must be >1");
  }

  let forceCpu = false;
  if (N * M * K < operationsGpuThreshold) {
    forceCpu = true;
  }

  // Initialize the result matrix with zeros.
  const result = Array.from({ length: N }, () => Array(K).fill(0));

  const numRowBlocks1 = Math.ceil(N / batchSize[0]);
  const numColBlocks1 = Math.ceil(M / batchSize[1]);
  const numColBlocks2 = Math.ceil(K / batchSize[1]);

  for (let i = 0; i < numRowBlocks1; i++) {
    for (let j = 0; j < numColBlocks2; j++) {
      for (let k = 0; k < numColBlocks1; k++) {
        
        // Calculate block indices (start and end rows/cols)
        const rowStart1 = i * batchSize[0];
        const rowEnd1 = Math.min((i + 1) * batchSize[0], N);
        const colStart1 = k * batchSize[1];
        const colEnd1 = Math.min((k + 1) * batchSize[1], M);

        const rowStart2 = k * batchSize[1];
        const rowEnd2 = Math.min((k + 1) * batchSize[1], M);
        const colStart2 = j * batchSize[1];
        const colEnd2 = Math.min((j + 1) * batchSize[1], K);
          
        // Multiply and accumulate
        const product =  await _matrixMultiplySlice(matrix1, matrix2, 
          [[rowStart1, rowEnd1],[colStart1, colEnd1]],[[rowStart2, rowEnd2],[colStart2, colEnd2]], forceCpu);

        for (let pRow = 0; pRow < product.length; pRow++) {
          for (let pCol = 0; pCol < product[0].length; pCol++) {
            result[rowStart1 + pRow][colStart2 + pCol] += product[pRow][pCol];
          }
        }
      }
    }
  }

  return result;
}

async function _matrixMultiplySlice(A, B, aSlice, bSlice, forceCpu = false) {
 const slicedARows = aSlice[0][1] - aSlice[0][0];
 const slicedACols = aSlice[1][1] - aSlice[1][0];
 const slicedBRows = bSlice[0][1] - bSlice[0][0];
 const slicedBCols = bSlice[1][1] - bSlice[1][0];

 if (forceCpu) {
  return _cpuMatrixMultiply(A, B, aSlice, bSlice);
} 

const gpuDevice = await getGpuDevice();
if (!gpuDevice) {
  console.warn("No GPU device available. Falling back to CPU multiplication.");
  return _cpuMatrixMultiply(A, B, aSlice, bSlice);
}


 // Try GPU multiplication in a try/catch
 try {
   // Flatten A, using the slice
   const Adata = new Float32Array(slicedARows * slicedACols);
   let idx = 0;
   for (let i = aSlice[0][0]; i < aSlice[0][1]; i++) {
     for (let j = aSlice[1][0]; j < aSlice[1][1]; j++) {
       Adata[idx++] = A[i][j];
     }
   }

   // Flatten B, using the slice
   const Bdata = new Float32Array(slicedBRows * slicedBCols);
   idx = 0;
   for (let i = bSlice[0][0]; i < bSlice[0][1]; i++) {
     for (let j = bSlice[1][0]; j < bSlice[1][1]; j++) {
       Bdata[idx++] = B[i][j];
     }
   }

   // Prepare buffers
   const bufferA = gpuDevice.createBuffer({
     size: Adata.byteLength,
     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
   });
   const bufferB = gpuDevice.createBuffer({
     size: Bdata.byteLength,
     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
   });

   const cRows = slicedARows;
   const cCols = slicedBCols;
   const cSizeBytes = 4 * cRows * cCols;

   const bufferC = gpuDevice.createBuffer({
     size: cSizeBytes,
     usage:
       GPUBufferUsage.STORAGE |
       GPUBufferUsage.COPY_SRC |
       GPUBufferUsage.COPY_DST
   });

   // Uniform buffer {slicedARows, slicedACols, slicedBCols, 0}
   const uniformBuffer = gpuDevice.createBuffer({
     size: 16, // 4 x 4 bytes
     usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
   });

   // Upload data
   gpuDevice.queue.writeBuffer(bufferA, 0, Adata);
   gpuDevice.queue.writeBuffer(bufferB, 0, Bdata);
   const dims = new Uint32Array([slicedARows, slicedACols, slicedBCols, 0]);
   gpuDevice.queue.writeBuffer(uniformBuffer, 0, dims);

   // WGSL shader
   const shaderCode = /* wgsl */ `
       @group(0) @binding(0) var<storage, read> A : array<f32>;
       @group(0) @binding(1) var<storage, read> B : array<f32>;
       @group(0) @binding(2) var<storage, read_write> C : array<f32>;
       @group(0) @binding(3) var<uniform> dims : vec4<u32>;
       // dims.x = slicedARows, dims.y = slicedACols, dims.z = slicedBCols

       @compute @workgroup_size(16, 16)
       fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
         let aRows = dims.x;
         let aCols = dims.y;
         let bCols = dims.z;

         let row = gid.y;
         let col = gid.x;

         if (row < aRows && col < bCols) {
           var sum = 0.0;
           for (var k = 0u; k < aCols; k++) {
             sum += A[row * aCols + k] * B[k * bCols + col];
           }
           C[row * bCols + col] = sum;
         }
       }
     `;
   const shaderModule = gpuDevice.createShaderModule({ code: shaderCode });

   // Pipeline
   const pipeline = gpuDevice.createComputePipeline({
     layout: "auto",
     compute: {
       module: shaderModule,
       entryPoint: "main"
     }
   });

   // Bind group
   const bindGroup = gpuDevice.createBindGroup({
     layout: pipeline.getBindGroupLayout(0),
     entries: [
       { binding: 0, resource: { buffer: bufferA } },
       { binding: 1, resource: { buffer: bufferB } },
       { binding: 2, resource: { buffer: bufferC } },
       { binding: 3, resource: { buffer: uniformBuffer } }
     ]
   });

   // Encode commands
   const commandEncoder = gpuDevice.createCommandEncoder();
   const passEncoder = commandEncoder.beginComputePass();
   passEncoder.setPipeline(pipeline);
   passEncoder.setBindGroup(0, bindGroup);

   // Dispatch
   const workgroupSize = 16;
   const dispatchX = Math.ceil(cCols / workgroupSize);
   const dispatchY = Math.ceil(cRows / workgroupSize);
   passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
   passEncoder.end();

   // Copy back to a CPU-readable buffer
   const readBuffer = gpuDevice.createBuffer({
     size: cSizeBytes,
     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
   });
   commandEncoder.copyBufferToBuffer(bufferC, 0, readBuffer, 0, cSizeBytes);

   // Submit
   gpuDevice.queue.submit([commandEncoder.finish()]);
   await readBuffer.mapAsync(GPUMapMode.READ);
   const arrBuffer = readBuffer.getMappedRange();
   const result = new Float32Array(arrBuffer.slice(0));
   readBuffer.unmap();

   // Convert to 2D
   const C = [];
   let pos = 0;
   for (let i = 0; i < cRows; i++) {
     const rowData = [];
     for (let j = 0; j < cCols; j++) {
       rowData.push(result[pos++]);
     }
     C.push(rowData);
   }

   return C;
 } catch (err) {
   console.error("WebGPU error:", err, "Falling back to CPU multiplication.");
   // If anything fails on GPU side, fallback:
   return _cpuMatrixMultiply(A, B, aSlice, bSlice);
 }
}

function _cpuMatrixMultiply(A, B, aSlice, bSlice) {
  const [aRowStart, aRowEnd] = aSlice[0];
  const [aColStart, aColEnd] = aSlice[1];
  const [bRowStart, _] = bSlice[0];
  const [bColStart, bColEnd] = bSlice[1];

  const aCols = aColEnd - aColStart;

  const resultRows = aRowEnd - aRowStart;
  const resultCols = bColEnd - bColStart;
  const C = Array(resultRows).fill(null).map(() => Array(resultCols).fill(0));

  for (let i = 0; i < resultRows; i++) {
    for (let j = 0; j < resultCols; j++) {
      for (let k = 0; k < aCols; k++) {
        C[i][j] += A[aRowStart + i][aColStart + k] * B[bRowStart + k][bColStart + j];
      }
    }
  }

  return C;
}