
#include <stdio.h>
#include <stdlib.h>
#include <helper_timer.h>

// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// + lst[n-1]}

#define BLOCK_SIZE 512 //@@ You can change this

__global__ void scan(int *input, int *output, int *aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  
  __shared__ float intermediateReduction[BLOCK_SIZE << 1];
    unsigned int tx = threadIdx.x;
	unsigned int initialElement = 2 * blockIdx.x * BLOCK_SIZE;
	unsigned int tempValueStorage=initialElement + tx;
	int strideIterator=0;
	int iterator = (tx + 1) * strideIterator * 2 - 1;
    
	if (tempValueStorage < len)
	{
       intermediateReduction[tx] = input[tempValueStorage];
    }
	else
	{
       intermediateReduction[tx] = 0;
    }
	if (tempValueStorage+ BLOCK_SIZE < len)
    {
		intermediateReduction[BLOCK_SIZE + tx] = input[tempValueStorage + BLOCK_SIZE];
    }
    else
	{
       intermediateReduction[BLOCK_SIZE + tx] = 0;
    }
	__syncthreads();

    
    for (strideIterator = 1; strideIterator <= BLOCK_SIZE; strideIterator <<= 1)
	{
	   iterator = (tx + 1) * strideIterator * 2 - 1;
       if (iterator < 2 * BLOCK_SIZE)
	   {
          intermediateReduction[iterator] += intermediateReduction[iterator - strideIterator];
	   }
       __syncthreads();
    }

    
    for (strideIterator = BLOCK_SIZE >> 1; strideIterator; strideIterator >>= 1) 
	{
	   int iterator = (tx + 1) * strideIterator * 2 - 1;
       if (iterator + strideIterator < 2 * BLOCK_SIZE)
	   {
          intermediateReduction[iterator + strideIterator] += intermediateReduction[iterator];
	   }
	   __syncthreads();
    }

    if (tempValueStorage < len)
       output[tempValueStorage] = intermediateReduction[tx];
    if (tempValueStorage + BLOCK_SIZE < len)
       output[tempValueStorage + BLOCK_SIZE] = intermediateReduction[BLOCK_SIZE + tx];

    if (aux && tx == 0)
       aux[blockIdx.x] = intermediateReduction[2 * BLOCK_SIZE - 1];
  
}
__global__ void finalReductionAddKernel(int *input, int *aux, int len) 
{
    unsigned int tx = threadIdx.x;
	unsigned int initialElement = 2 * blockIdx.x * BLOCK_SIZE;
	unsigned int tempValueStorage=initialElement + tx;
	
    if (blockIdx.x) 
	{
       if (tempValueStorage + BLOCK_SIZE < len)
       input[tempValueStorage + BLOCK_SIZE] =input[tempValueStorage + BLOCK_SIZE] + aux[blockIdx.x];
	   if (tempValueStorage < len)
       input[tempValueStorage] = input[tempValueStorage] + aux[blockIdx.x ];
    }
}

int main(int argc, char **argv) {
  int *hostInput;  // The input 1D list
  int *hostOutput; // The output list
  int *expectedOutput;
  int *deviceInput;
  int *deviceOutput;
  int *deviceAuxArray, *deviceAuxScannedArray;
  int numElements; // number of elements in the list
  
  FILE *infile, *outfile;
  int inputLength, outputLength;
  StopWatchLinux stw;
  unsigned int blog = 1;

  // Import host input data
  stw.start();
  if ((infile = fopen("input.raw", "r")) == NULL)
  { printf("Cannot open input.raw.\n"); exit(EXIT_FAILURE); }
  fscanf(infile, "%i", &inputLength);
  hostInput = (int *)malloc(sizeof(int) * inputLength);
  for (int i = 0; i < inputLength; i++)
     fscanf(infile, "%i", &hostInput[i]);
  fclose(infile);
  inputLength = inputLength;
  hostOutput = (int *)malloc(inputLength * sizeof(int));
  stw.stop();
  printf("Importing data and creating memory on host: %f ms\n", stw.getTime());

  if (blog) printf("*** The number of input elements in the input is %i\n", inputLength);

  stw.reset();
  stw.start();
  
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(int));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(int));

  cudaMalloc(&deviceAuxArray, (BLOCK_SIZE << 1) * sizeof(int));
  cudaMalloc(&deviceAuxScannedArray, (BLOCK_SIZE << 1) * sizeof(int));
  
  stw.stop();
  printf("Allocating GPU memory: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  cudaMemset(deviceOutput, 0, inputLength * sizeof(int));
  
  stw.stop();
  printf("Clearing output memory: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(int),cudaMemcpyHostToDevice);

  stw.stop();
  printf("Copying input memory to the GPU: %f ms\n", stw.getTime());

  //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil((float)inputLength/(BLOCK_SIZE<<1)), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

  stw.reset();
  stw.start();
  
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device
  scan<<<dimGrid, dimBlock>>>(deviceInput,deviceOutput,deviceAuxArray,inputLength);
  scan<<<dim3(1,1,1), dimBlock>>>(deviceAuxArray, deviceAuxScannedArray, NULL, BLOCK_SIZE << 1);
  finalReductionAddKernel<<<dimGrid, dimBlock>>>(deviceOutput, deviceAuxScannedArray, inputLength);

  cudaDeviceSynchronize();
 
  stw.stop();
  printf("Performing CUDA computation: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(int),cudaMemcpyDeviceToHost);
  
  stw.stop();
  printf("Copying output memory to the CPU: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray);
  cudaFree(deviceAuxScannedArray);

  stw.stop();
  printf("Freeing GPU Memory: %f ms\n", stw.getTime());

  if ((outfile = fopen("output.raw", "r")) == NULL)
  { printf("Cannot open output.raw.\n"); exit(EXIT_FAILURE); }
  fscanf(outfile, "%i", &outputLength);
  expectedOutput = (int *)malloc(sizeof(int) * outputLength);  
  for (int i = 0; i < outputLength; i++)
     fscanf(outfile, "%i", &expectedOutput[i]);	
  fclose(outfile);
  
  int test = 1;
for (int i=0;i<outputLength;i++)
  {
      printf("%i\n",hostOutput[i]);
  }
  for (int i = 0; i < outputLength; i++) {
     if (expectedOutput[i] != hostOutput[i])
        printf("%i %i %i\n", i, expectedOutput[i], hostOutput[i]);
     test = test && (expectedOutput[i] == hostOutput[i]);
  }
  
  if (test) printf("Results correct.\n");
  else printf("Results incorrect.\n");

  free(hostInput);
  cudaFreeHost(hostOutput);
  free(expectedOutput);

  return 0;
}
