/* // Compilation instructions
 * mex -v FMapNLSumT_gpu.cu -largeArrayDims
 * CFLAGS="\$CFLAGS -std=c++11 -g" LDFLAGS='\$LDFLAGS -Wl,-rpath,/usr/local/cuda/lib'
 * -I/usr/local/include/
 * -L/usr/local/cuda/lib -lcudart -lstdc++ -lc */


/* Let us assume that X is of size (H * W) x D x N, idx is of size 
 * (H * W) x Nb x N (computed by misc.patchMatch) where 
 * the elements of idx have values in the range [1, H*W] and Weights is of 
 * size (H x W) x (D x Nb) x N or (H x W) x Nb x N or 
 * D x Nb or Nb. In the second case the same weights are applied on the 
 * D feature maps at each spatial dimension, while in the last two cases  
 * the same weights are applied to the H x W spatial elements.
 *
 * Then the output Y will be of size (H * W) x D x N where 
 *
 *            Nb
 * Y(i,d,n) = Sum W(i,idx_w(d,r),n)*X(idx(i,r,n),d,n), i=1:H*W, d=1:D, n=1:N
 *            r=1
 * and idx_w(d,r) = r*D + d
 *
 *          Nb
 * Y(ind_y) = Sum W(ind_w)*X(ind_x) where
 *          r=1
 * ind_y = i + d*H*W + n*H*W*D, 
 * ind_w = i + (d+r*D)*H*W + n*H*W*D*Nb,  (W is of size HW x D x Nb x N)
 * ind_x = idx[ind_r] + d*H*W + n*H*W*D where 
 * ind_r = i + r*H*W + n*H*W*Nb
 *
 * To check the correctness of the computation do the following in Matlab:
 * 
 * H=10;W=20;D=3;N=2;Nb=4;
 * X=randn(H,W,D,N); idx=randsrc(H*W*Nb*N,1,[1:H*W]); idx = reshape(idx,H,W,Nb,N);
 * idx(:,:,1,1) = reshape(1:H*W,H,W); idx(:,:,1,2)=idx(:,:,1,1); idx = uint32(idx);
 * Weights = randn(H,W,Nb,N);
 * [X,idx,Weights] = misc.move_data('gpu',X,idx,Weights);
 *
 *
 * Y = zeros(size(X),'like',X);
 * for i = 1:H, for j=1:W, for d=1:D, for n=1:N, for r=1:Nb, 
 * Y(i,j,d,n) = Y(i,j,d,n) + Weights(i,j,r,n)*X(idx(i,j,r,n)+(d-1)*H*W+(n-1)*H*W*D);
 * end,end,end,end,end
 *
 * Y2 = FMapNLSum_gpu(X,Weights,idx-1); % We substract 1 from idx since
 * % in C++ the starting element is zero and not 1.
 *
 * e = Y-Y2; max(e(:)), min(e(:))
 *
 * 
 * Now in order to compute the adjoint operation let us consider the following
 * example where H*W = 6, D=3, Nb = 2 and N = 1
 *
 * Let X = [ x1:D    idx = [1 3 4   Weights = [ W1:D,1 W1:D,2 W1:D,3
 *           x2:D           2 1 6               W2:D,1 W2:D,2 W2:D,3
 *           x3:D           3 5 2               W3:D,1 W3:D,2 W3:D,3
 *           x4:D           4 1 6               W4:D,1 W4:D,2 W4:D,3
 *           x5:D           5 2 3               W5:D,1 W5:D,2 W5:D,3
 *           x6:D ]         6 4 3]              W6:D,1 W6:D,2 W6:D,3]
 *
 *  xk:D ==> It denotes all the D channels for the k spatial coordinate
 *  W(k,r):D ==> It denotes the D-dimensional weight vector for the rth 
 *  closest neighbor.
 *        
 *  Based on the above, the forward operation is given by:
 *
 *  Y = [ x1:D .* W1:D,1 + x3:D .* W1:D,2 + x4:D .* W1:D,3
 *        x2:D .* W2:D,1 + x1:D .* W2:D,2 + x6:D .* W2:D,3
 *        x3:D .* W3:D,1 + x5:D .* W3:D,2 + x2:D .* W3:D,3
 *        x4:D .* W4:D,1 + x1:D .* W4:D,2 + x6:D .* W4:D,3
 *        x5:D .* W5:D,1 + x2:D .* W5:D,2 + x3:D .* W5:D,3
 *        x6:D .* W6:D,1 + x4:D .* W6:D,2 + x3:D .* W6:D,3 ]
 *
 *  and the adjoint is given by
 *
 *  Z = [ y1:D .* W1:D,1 + y2:D .* W2:D,2 + y4:D .* W4:D,2
 *        y2:D .* W2:D,1 + y3:D .* W3:D,3 + y5:D .* W5:D,2
 *        y1:D .* W1:D,2 + y3:D .* W3:D,1 + y5:D .* W5:D,3 + y6:D .* W6:D,3
 *        y1:D .* W1:D,3 + y4:D .* W4:D,1 + y6:D .* W6:D,2
 *        y2:D .* W2:D,2 + y5:D .* W5:D,1
 *        y2:D .* W2:D,3 + y4:D .* W4:D,3 + y6:D .* W6:D,1 ]
 *
 *  We observe that zk:D,n is given as the weighted sum of Wl:D,m,n and yl:D,n 
 *  where l,m,n are the row, column and slice where the number k+N*n is 
 *  found in idx. (In this example we considered N=1 but this holds true
 *  for N >= 1). 
 *  
 *  NOTE!!!!
 *  In order the adjoint to correctly work for multiple images 
 *  we must transform the idx computed by misc.patchMatch so that 
 *  idx_new(:,:,n) = idx(:,:,n) + (n-1)*H*W. This is taken care by the 
 *  surrogate function FMapNLSumT_helper.m
 *
 *  [widx,n,I]=misc.FMapNLSumT_helper(idx);
 *  Z = FMapNLSumT_gpu(Y,Weights,widx-1,n,I-1);
 *  e = Z(:)'* X(:) - Y(:)'*Y(:) // This should be close to zero.
 * /

/* In a mxArray to access the element X[i][j][z] you can do it by referring
   to the element X[i+j*dims[0]+z*dims[0]*dims[1]] */


#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda.h>
#include <cuda_runtime.h>

#if __CUDA_ARCH__ >= 200
#define VL_CUDA_NUM_THREADS 1024
#else
#define VL_CUDA_NUM_THREADS 512
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line)
{
  if (code != cudaSuccess)
  {
    char *err_str = new char[1000];
    sprintf(err_str,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    mexErrMsgTxt(err_str);
    delete [] err_str;
  }
}

inline unsigned int divideUpwards(unsigned int a, unsigned int b)
{
  return (a + b - 1) / b ;
}

__device__ void index_map(const mwSize *c, const mwSize index, mwSize *ind){
  
  // From ndx = i + j*H + d*H*W + n*H*W*D and c = [H*W, H*W*D, Nb]
  // we want to recover ind[0] = i + j*H, ind[1]=d and ind[2]=n
  
  ind[0] = index % c[0]; // i+j*H
  ind[1] = ((index - ind[0]) % c[1]) / c[0]; // d
  ind[2] = index / c[1]; // n
  
}


template <typename T>
        __global__ void FMapNLSumT_gpu_kernel(
        const T *X, const T *W, T* Y,
        const unsigned int *n_table,
        const unsigned int *I,
        const unsigned int *idx,
        const mwSize *c,
        const bool FMapSharedWeights,
        const bool SpatialSharedWeights,
        size_t numElements)
{
  
  size_t index = static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  
  if (index < numElements){
    mwSize ind[3];
    mwSize coord, coord_X, coord_W;
    index_map(c,index,ind);
    // For any channel k of X(i,j,k,n) we have to use n_table(i,j,n) and
    // I(i,j,n).
    // index_n = i+j*H+n*H*W
    mwSize index_n = ind[0]+c[0]*ind[2];
  
    for (mwSize k = 0; k < n_table[index_n]; ++k){
    
      coord = idx[I[index_n]+k]; // coord = i + j*H + r*H*W + n*H*W*Nb
      coord_X = (coord % c[0])+c[0]*ind[1]+c[1]*ind[2]; // coord_X = i + j*H + d*H*W + n*H*W*N
    
      if (!FMapSharedWeights && !SpatialSharedWeights){
        // coord_W  = i + j*H + d*H*W + r*H*W*D + n*H*W*D*Nb
        coord_W = (coord % c[0]) + c[0]*ind[1] + c[1]*((coord / c[0]) % c[2]) + ind[2]*c[1]*c[2];
      }
      else if (FMapSharedWeights && !SpatialSharedWeights){
        // coord_W = i + j*H + r*H*W + n*H*W*Nb
        coord_W = coord;
      }
      else if (!FMapSharedWeights && SpatialSharedWeights){
        // coord_W = d + D*r
        coord_W = ind[1]+(c[1]/c[0])*((coord / c[0]) % c[2]);
      }
      else{
        // coord_W = r
        coord_W = (coord / c[0]) % c[2];
      }
    
      Y[index] += X[coord_X]*W[coord_W];
    }       
  }
}


template <typename T> 
static inline cudaError_t FMapNLSumT_gpu(
        const T *X, const T *W, T *Y, 
        const unsigned int *n_table,
        const unsigned int *I,
        const unsigned int *idx,
        const mwSize *c,
        const bool FMapSharedWeights,
        const bool SpatialSharedWeights,        
        size_t numElements)
{  
 FMapNLSumT_gpu_kernel<T>
 <<< divideUpwards(numElements, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
 (X,W,Y,n_table,I,idx,c,FMapSharedWeights,SpatialSharedWeights,numElements);
 return cudaPeekAtLastError(); 
}


void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, mxArray const *prhs[])
{
  
  // Create Error Messages.
  char const * const errId = "FMapNLSumT_gpu:InvalidInput";
  char const * const errMsg_narg = "Invalid input: Five input "
          "arguments are expected.";
  char const * const errMsg_ty = "Invalid input: Input has unsupported type.";
  char const * const errMsg_args = "Invalid input: All the inputs must be between a 1-D to 4-D array.";
  char const * const errMsg_dims = "Invalid input: Dimensions mismatch.";
  char const * const errMsg_type = "Invalid input: Inputs 1, 2 must be of the same data type.";
  char const * const errMsg_type2 = "Invalid input: Inputs 3-5 must be of type 'uint32'.";
  
  
  /* Initialize the MathWorks GPU API. */
  mxInitGPU(); 
  
  if (nrhs != 5)
    mexErrMsgIdAndTxt(errId, errMsg_narg);
  
  // Get the inputs
  // X 4D input [H W D N] where D is the number of channels and N the
  // number of images.
  const mxGPUArray *X_mx = mxGPUCreateFromMxArray(prhs[0]);
  // 4D array [H W (D*Nb | Nb) N] or 1D array [D*Nb | Nb]
  const mxGPUArray *Weights_mx  = mxGPUCreateFromMxArray(prhs[1]);
  // 1D array [H*W*Nb*N]
  const mxGPUArray *idx_mx  = mxGPUCreateFromMxArray(prhs[2]);
  // 1D array [H*W*N]
  const mxGPUArray *n_mx  = mxGPUCreateFromMxArray(prhs[3]);
  // 1D array [H*W*N]
  const mxGPUArray *I_mx  = mxGPUCreateFromMxArray(prhs[4]);
  
  const mxClassID cid = mxGPUGetClassID(X_mx);
  if (cid != mxGPUGetClassID(Weights_mx))
    mexErrMsgIdAndTxt(errId,errMsg_type);
  
  const mxClassID cid2 = mxGPUGetClassID(idx_mx);
  if (cid2 != mxUINT32_CLASS)
    mexErrMsgIdAndTxt(errId,errMsg_type2);
  
  if (cid2 != mxGPUGetClassID(n_mx) || cid2 != mxGPUGetClassID(I_mx))
    mexErrMsgIdAndTxt(errId,errMsg_type2);
  
  // X 4D Array [H W D N]
  const mwSize X_ndims = mxGPUGetNumberOfDimensions(X_mx);
  if ( X_ndims < 1 || X_ndims > 4)
    mexErrMsgIdAndTxt(errId, errMsg_args);
  
  const mwSize *X_dims = mxGPUGetDimensions(X_mx);
  const mwSize H = X_dims[0];
  const mwSize W = (X_ndims < 2) ? 1 : X_dims[1];
  const mwSize D = (X_ndims < 3) ? 1 : X_dims[2];
  const mwSize N = (X_ndims < 4) ? 1 : X_dims[3];
  
  mwSize numElements=H*W*D*N;
  
  // idx 1D Array [H*W*Nb*N]
  mwSize idxElements = mxGPUGetNumberOfElements(idx_mx);
  const mwSize Nb = idxElements/(H*W*N);
  if (idxElements % (H*W*N) != 0)
    mexErrMsgIdAndTxt(errId,errMsg_dims);
  const unsigned int *idx_ptr = static_cast<const unsigned int*>(mxGPUGetDataReadOnly(idx_mx));
  
  // Weights : 4D Array [H W D*Nb N] or [H W Nb N] or 2D Array [D*Nb] or [Nb]
  const mwSize *Weights_dims = mxGPUGetDimensions(Weights_mx);
  const mwSize W_ndims = mxGPUGetNumberOfDimensions(Weights_mx);
  if ( W_ndims < 1 || W_ndims > 4)
    mexErrMsgIdAndTxt(errId, errMsg_args);
  
  const mwSize numWeightElements = mxGPUGetNumberOfElements(Weights_mx);
    
  bool SpatialSharedWeights = false; 
  bool FMapSharedWeights = false;  
  if (W_ndims == 2){
    SpatialSharedWeights = true; // Weights 1D Array of D*Nb or Nb elements
    
    if (numWeightElements == Nb)
      FMapSharedWeights = true;
    
    if (!FMapSharedWeights && numWeightElements != Nb*D)
      mexErrMsgIdAndTxt(errId, errMsg_dims);    
  }
  
  if (W_ndims > 2){
    if ((Weights_dims[0] != H) || (Weights_dims[1] != W))
      mexErrMsgIdAndTxt(errId, errMsg_dims);
    
    if (Weights_dims[2] == Nb)
      FMapSharedWeights = true;
    
    if (!FMapSharedWeights && Weights_dims[2] != Nb*D)
      mexErrMsgIdAndTxt(errId, errMsg_dims);
    
    if (W_ndims < 4){
      if (N != 1)
        mexErrMsgIdAndTxt(errId, errMsg_dims);
    }
    else{
      if (Weights_dims[3] != N)
        mexErrMsgIdAndTxt(errId, errMsg_dims);
    }
  }  
  
  // n 1D Array [H*W*N]
  if (mxGPUGetNumberOfElements(n_mx)!=H*W*N)
    mexErrMsgIdAndTxt(errId,errMsg_dims);
  const unsigned int *n_ptr = static_cast<const unsigned int*>(mxGPUGetDataReadOnly(n_mx));
  
  // I 1D Array [H*W*N]
  if (mxGPUGetNumberOfElements(I_mx)!=H*W*N)
    mexErrMsgIdAndTxt(errId,errMsg_dims);
  const unsigned int *I_ptr = static_cast<const unsigned int*>(mxGPUGetDataReadOnly(I_mx));
  
  
  // Create output
  
  // Y [H, W, D, N]
  const mwSize Y_dims[]={H, W, D, N};
  mxGPUArray *Y_mx = mxGPUCreateGPUArray(4, Y_dims, cid, mxREAL, MX_GPU_INITIALIZE_VALUES);
  
  const mwSize c_ptr[3]={H*W,H*W*D,Nb};
  mwSize *d_c;
  cudaMalloc(&d_c,3*sizeof(mwSize));
  cudaMemcpy(d_c,c_ptr,3*sizeof(mwSize),cudaMemcpyHostToDevice);
  
  if (cid == mxDOUBLE_CLASS){
    const double *X_ptr = static_cast<const double*>(mxGPUGetDataReadOnly(X_mx));
    const double *Weights_ptr = static_cast<const double*>(mxGPUGetDataReadOnly(Weights_mx));
    double *Y_ptr = static_cast<double*>(mxGPUGetData(Y_mx));
    
    gpuErrchk(FMapNLSumT_gpu<double>(X_ptr,Weights_ptr,Y_ptr,n_ptr,
    I_ptr,idx_ptr,d_c,FMapSharedWeights,SpatialSharedWeights,numElements));
  }
  else if (cid == mxSINGLE_CLASS){
    const float *X_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(X_mx));
    const float *Weights_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(Weights_mx));
    float *Y_ptr = static_cast<float*>(mxGPUGetData(Y_mx));
    
    gpuErrchk(FMapNLSumT_gpu<float>(X_ptr,Weights_ptr,Y_ptr,n_ptr,
    I_ptr,idx_ptr,d_c,FMapSharedWeights,SpatialSharedWeights,numElements));    
  }
  else
    mexErrMsgIdAndTxt(errId, errMsg_ty);
  
  
  plhs[0] = mxGPUCreateMxArrayOnGPU(Y_mx);
  
  mxGPUDestroyGPUArray(X_mx);
  mxGPUDestroyGPUArray(Weights_mx);
  mxGPUDestroyGPUArray(Y_mx);
  mxGPUDestroyGPUArray(idx_mx);
  mxGPUDestroyGPUArray(n_mx);
  mxGPUDestroyGPUArray(I_mx);  
}










