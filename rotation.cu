#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "cpu_bitmap.h"
#include "bitmap_help.h"

#define CHANNELS 4

__global__ void Rotate(float* Source, float* Destination, int sizeX, int sizeY, float deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int xc = sizeX - sizeX/2;
    int yc = sizeY - sizeY/2;
    int newx = ((float)i-xc)*cos(deg) - ((float)j-yc)*sin(deg) + xc;
    int newy = ((float)i-xc)*sin(deg) + ((float)j-yc)*cos(deg) + yc;
    if (newx >= 0 && newx < sizeX && newy >= 0 && newy < sizeY)
    {
        putPixVal(Destination, sizeX, i , j, readPixVal(Source, sizeX, newx, newy));
    }
}


__device__ float readPixVal( float* ImgSrc,int ImgWidth,int x,int y)
{
    return (float)ImgSrc[y*ImgWidth+x];
}

__device__ void putPixVal( float* ImgSrc,int ImgWidth,int x,int y, float floatVal)
{
    ImgSrc[y*ImgWidth+x] = floatVal;
}


__host__ void imgProc(unsigned char * map, int size, int width, int height) {

	unsigned char* device_image;
    
	size_t imageSize= size;
	float deg=90;

   cudaMalloc((void**)&device_image,imageSize);
   cudaMemcpy(device_image,map,imageSize,cudaMemcpyHostToDevice);

   dim3 gridSize(width,height);
   dim3 blockSize(1,1,1);
   Rotate<<<gridSize,blockSize>>>(map,device_image, width, height,deg);
   cudaDeviceSynchronize();
   cudaMemcpy(map,device_image,imageSize,cudaMemcpyDeviceToHost);
   
   cudaFree(device_image);
   return;
}


int main(void) {
   char fname[50];
   FILE* infile;
   unsigned short ftype;
   tagBMFH bitHead;
   tagBMIH bitInfoHead;
   tagRGBQ *pRgb;

   printf("Please enter the .bmp file name: ");
   scanf("%s", fname);
   strcat(fname,".bmp");
   infile = fopen(fname, "rb");

   if (infile != NULL) {
      printf("File open successful.\n");
      fread(&ftype, 1, sizeof(unsigned short), infile);
      if (ftype != 0x4d42)
      {
         printf("File not .bmp format.\n");
         return 1;
      }
      fread(&bitHead, 1, sizeof(tagBMFH), infile);
      fread(&bitInfoHead, 1, sizeof(tagBMIH), infile);      
   }
   else {
      printf("File open fail.\n");
      return 1;
   }

   if (bitInfoHead.biBitCount < 24) {
      long nPlateNum = long(pow(2, double(bitInfoHead.biBitCount)));
      pRgb = (tagRGBQ *)malloc(nPlateNum * sizeof(tagRGBQ));
      memset(pRgb, 0, nPlateNum * sizeof(tagRGBQ));
      int num = fread(pRgb, 4, nPlateNum, infile);
   }

   int width = bitInfoHead.biWidth;
   int height = bitInfoHead.biHeight;
   int l_width = 4 * ((width * bitInfoHead.biBitCount + 31) / 32);
   long nData = height * l_width;
   unsigned char *pColorData = (unsigned char *)malloc(nData);
   memset(pColorData, 0, nData);
   fread(pColorData, 1, nData, infile);

   fclose(infile);
   
   CPUBitmap dataOfBmp(width, height);
   unsigned char *map = dataOfBmp.get_ptr();

   if (bitInfoHead.biBitCount < 24) {
      int k, index = 0;
      if (bitInfoHead.biBitCount == 1) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j / 8;
               mixIndex = pColorData[k];
               if (j % 8 < 7) mixIndex = mixIndex << (7 - (j % 8));
               mixIndex = mixIndex >> 7;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
       else if (bitInfoHead.biBitCount == 2) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j / 4;
               mixIndex = pColorData[k];
               if (j % 4 < 3) mixIndex = mixIndex << (6 - 2 * (j % 4));
               mixIndex = mixIndex >> 6;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
       else if (bitInfoHead.biBitCount == 4) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j / 2;
               mixIndex = pColorData[k];
               if (j % 2 == 0) mixIndex = mixIndex << 4;
               mixIndex = mixIndex >> 4;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
       else if (bitInfoHead.biBitCount == 8) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j;
               mixIndex = pColorData[k];
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
       else if (bitInfoHead.biBitCount == 16) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j * 2;
               unsigned char shortTemp = pColorData[k + 1] << 8;
               mixIndex = pColorData[k] + shortTemp;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
   }
   else {
      int k, index = 0;
      for (int i = 0; i < height; i++)
         for (int j = 0; j < width; j++) {
            k = i * l_width + j * 3;
            map[index * 4 + 0] = pColorData[k + 2];
            map[index * 4 + 1] = pColorData[k + 1];
            map[index * 4 + 2] = pColorData[k];
            index++;
         }
   }

   imgProc(map, dataOfBmp.image_size(), width, height);
   dataOfBmp.display_and_exit();
   return 0;
}