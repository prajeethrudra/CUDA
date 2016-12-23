#include <stdio.h>
#include <stdlib.h>
#include <helper_timer.h>
#include <cuda.h>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

  //@@ Declare and clear privatized bins

  int tx = threadIdx.x + blockDim.x * blockIdx.x ;
  __shared__ unsigned int privateHistogramBin[4096];

  for(int i=0; i<num_bins;i++)
  {
	privateHistogramBin[i] = 0;
  }
  __syncthreads(); 

  //@@ Compute histogram
  if(tx<=num_elements)
  {
       atomicAdd((&privateHistogramBin[input[tx]]), 1);
  }
  __syncthreads();

  
  //@@ Commit to global memory

  for (int i = 0;i < (num_bins / 1024);i++)
	{
		int indexForDeviceBin =i * 1024 + tx;
		if (indexForDeviceBin < (num_bins))
		{
			atomicAdd(&(bins[indexForDeviceBin]), privateHistogramBin[indexForDeviceBin]);
		}
	}     
    __syncthreads();     
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
  //@@ Ensure bin values are not too large
   int i = threadIdx.x + blockDim.x * blockIdx.x ;
   if (bins[i] > 127) 
   {
	bins[i] = 127;
   }
   __syncthreads();
     
  }

void histogram(unsigned int *input, unsigned int *bins,
               unsigned int num_elements, unsigned int num_bins) {

  //@@ zero out bins
  cudaMemset(bins,0, num_bins * sizeof(unsigned int));

  //@@ Launch histogram_kernel on the bins
  histogram_kernel<<<1,1024>>>(input,bins,num_elements, num_bins);
  cudaDeviceSynchronize();
  
  //@@ Launch convert_kernel on the bins
  convert_kernel<<<1,1024>>>(bins,num_bins);
  cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
  int inputLength, outputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *expectedOutput;
  unsigned int *deviceInput;
  unsigned int *deviceBins;
  
  FILE *infile, *outfile;
  StopWatchLinux stw;
  unsigned int blog = 1;

  // Import host input data
  stw.start();
  if ((infile = fopen("input.raw", "r")) == NULL)
  { printf("Cannot open input.raw.\n"); exit(EXIT_FAILURE); }
  fscanf(infile, "%i", &inputLength);
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  hostInput = (unsigned int *)malloc(sizeof(unsigned int) * inputLength); 
  for (int i = 0; i < inputLength; i++)
     fscanf(infile, "%i", &hostInput[i]);
  fclose(infile);
  stw.stop();
  printf("Importing data and creating memory on host: %f ms\n", stw.getTime());

  if (blog) printf("*** The input length is %i\n", inputLength);
  if (blog) printf("*** The number of bins is %i\n", NUM_BINS);

  stw.reset();
  stw.start();
  
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput, sizeof(unsigned int)*inputLength );
  cudaMalloc((void **)&deviceBins, sizeof(unsigned int)*NUM_BINS);

  cudaDeviceSynchronize();
  
  
  stw.stop();
  printf("Allocating GPU memory: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  //@@ Copy memory to the GPU here
  
  cudaMemcpy(deviceInput,hostInput , sizeof(unsigned int) * inputLength , cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  
  stw.stop();
  printf("Copying input memory to the GPU: %f ms\n", stw.getTime());

  // Launch kernel
  // ----------------------------------------------------------
  if (blog) printf("*** Launching kernel");
  
  stw.reset();
  stw.start();

  histogram(deviceInput, deviceBins, inputLength, NUM_BINS);
  
  stw.stop();
  printf("Performing CUDA computation: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int) * NUM_BINS , cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
 
  stw.stop();
  printf("Copying output memory to the CPU: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  
  stw.stop();
  printf("Freeing GPU Memory: %f ms\n", stw.getTime());

  // Verify correctness
  // -----------------------------------------------------
  
  if ((outfile = fopen("output.raw", "r")) == NULL)
  { printf("Cannot open output.raw.\n"); exit(EXIT_FAILURE); }
  fscanf(outfile, "%i", &outputLength);
  expectedOutput = (unsigned int *)malloc(sizeof(unsigned int) * outputLength);  
  for (int i = 0; i < outputLength; i++)
     fscanf(outfile, "%i", &expectedOutput[i]);	
  fclose(outfile);
  
  int test = 1;
  for (int i = 0; i < outputLength; i++)
     test = test && (expectedOutput[i] == hostBins[i]);
  
  if (test) printf("Results correct.\n");
  else printf("Results incorrect.\n");

  free(hostBins);
  free(hostInput);
  free(expectedOutput);
  return 0;
}