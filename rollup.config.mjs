import terser from '@rollup/plugin-terser';

export default [
  {
    input: 'src/index.js', 
    output: {
      file: 'dist/matmul-gpu.js', 
      format: 'esm',
      name: 'matmul' 
    }
  },
  {
    input: 'src/index.js',
    output: {
      file: 'dist/matmul-gpu.umd.min.js',
      format: 'umd',
      name: 'matmul',
      sourcemap: true,
    },
    plugins: [
      terser()
    ]
  }
];