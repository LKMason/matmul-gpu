/**
 * Multiplies two 2D matrices, optionally using batched processing.
 * Uses WebGPU if available; otherwise falls back on a JS implementation.
 *
 * @async
 * @function matrixMultiply2D
 * @param {number[][]} matrix1 The first matrix (N x M).
 * @param {number[][]} matrix2 The second matrix (M x K).
 * @param {number} [batchSize=null] The size of the batches to use for processing.  If null (default),
 *  a batch size is chosen to be the maximum dimension of the input matrices, effectively resulting
 *  in no batching. Must be greater than 1 if specified.
 * @returns {number[][]} The resulting matrix (N x K).
 * @throws {Error} If the inner dimensions of the matrices do not match.
 * @throws {Error} If the batch size is less than 2.
 */
async function matrixMultiply(matrix1, matrix2, batchSize=1024) {
  const N = matrix1.length;
  const M = matrix1[0].length;
  const M2 = matrix2.length;
  const K = matrix2[0].length;

  if (M !== M2) {
    throw new Error("Inner dimensions of matrices must match for multiplication");
  }

  if (batchSize == null) {
    batchSize = Math.max(N, M, K);
  }

  if (batchSize < 2) {
    throw new Error("Batch size must be >1");
  }

  // Initialize the result matrix with zeros.
  const result = Array.from({ length: N }, () => Array(K).fill(0));

  const numRowBlocks1 = Math.ceil(N / batchSize);
  const numColBlocks1 = Math.ceil(M / batchSize);
  const numColBlocks2 = Math.ceil(K / batchSize);

  for (let i = 0; i < numRowBlocks1; i++) {
    for (let j = 0; j < numColBlocks2; j++) {
      for (let k = 0; k < numColBlocks1; k++) {
        
        // Calculate block indices (start and end rows/cols)
        const rowStart1 = i * batchSize;
        const rowEnd1 = Math.min((i + 1) * batchSize, N);
        const colStart1 = k * batchSize;
        const colEnd1 = Math.min((k + 1) * batchSize, M);

        const rowStart2 = k * batchSize;
        const rowEnd2 = Math.min((k + 1) * batchSize, M);
        const colStart2 = j * batchSize;
        const colEnd2 = Math.min((j + 1) * batchSize, K);

          
        // Multiply and accumulate
        const product =  await _matrixMultiplySlice(matrix1, matrix2, 
          [[rowStart1, rowEnd1],[colStart1, colEnd1]],[[rowStart2, rowEnd2],[colStart2, colEnd2]]);

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

async function _matrixMultiplySlice(A, B, aSlice, bSlice) {
 const slicedARows = aSlice[0][1] - aSlice[0][0];
 const slicedACols = aSlice[1][1] - aSlice[1][0];
 const slicedBRows = bSlice[0][1] - bSlice[0][0];
 const slicedBCols = bSlice[1][1] - bSlice[1][0];

 // Check WebGPU availability
 if (!("gpu" in navigator)) {
   console.error("WebGPU not supported. Falling back to CPU multiplication.");
   return _cpuMatrixMultiply(A, B, aSlice, bSlice);
 }

 // Try GPU multiplication in a try/catch
 try {
   const adapter = await navigator.gpu.requestAdapter();
   const device = await adapter.requestDevice();

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
   const bufferA = device.createBuffer({
     size: Adata.byteLength,
     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
   });
   const bufferB = device.createBuffer({
     size: Bdata.byteLength,
     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
   });

   const cRows = slicedARows;
   const cCols = slicedBCols;
   const cSizeBytes = 4 * cRows * cCols;

   const bufferC = device.createBuffer({
     size: cSizeBytes,
     usage:
       GPUBufferUsage.STORAGE |
       GPUBufferUsage.COPY_SRC |
       GPUBufferUsage.COPY_DST
   });

   // Uniform buffer {slicedARows, slicedACols, slicedBCols, 0}
   const uniformBuffer = device.createBuffer({
     size: 16, // 4 x 4 bytes
     usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
   });

   // Upload data
   device.queue.writeBuffer(bufferA, 0, Adata);
   device.queue.writeBuffer(bufferB, 0, Bdata);
   const dims = new Uint32Array([slicedARows, slicedACols, slicedBCols, 0]);
   device.queue.writeBuffer(uniformBuffer, 0, dims);

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
   const shaderModule = device.createShaderModule({ code: shaderCode });

   // Pipeline
   const pipeline = device.createComputePipeline({
     layout: "auto",
     compute: {
       module: shaderModule,
       entryPoint: "main"
     }
   });

   // Bind group
   const bindGroup = device.createBindGroup({
     layout: pipeline.getBindGroupLayout(0),
     entries: [
       { binding: 0, resource: { buffer: bufferA } },
       { binding: 1, resource: { buffer: bufferB } },
       { binding: 2, resource: { buffer: bufferC } },
       { binding: 3, resource: { buffer: uniformBuffer } }
     ]
   });

   // Encode commands
   const commandEncoder = device.createCommandEncoder();
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
   const readBuffer = device.createBuffer({
     size: cSizeBytes,
     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
   });
   commandEncoder.copyBufferToBuffer(bufferC, 0, readBuffer, 0, cSizeBytes);

   // Submit
   device.queue.submit([commandEncoder.finish()]);
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

export { matrixMultiply };
