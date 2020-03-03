// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Implements the GPU kernels used by the DFT algorithm.

#include "dca/phys/dca_step/cluster_solver/ctaux/accumulator/tp/kernels_interface_for_eqtime.hpp"
//#include "dca/phys/dca_step/cluster_solver/ctaux/accumulator/tp/singleton_obj_dev.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>


#include "dca/util/integer_division.hpp"
#include "dca/linalg/util/cast_cuda.hpp"
#include "dca/linalg/util/atomic_add_cuda.cu.hpp"
#include "dca/linalg/util/complex_operators_cuda.cu.hpp"
#include "dca/linalg/util/error_cuda.hpp"
#include "dca/phys/dca_step/cluster_solver/ctaux/accumulator/tp/TpEqTime_helper.cuh"
#include "dca/linalg/matrix.hpp"
#include "dca/linalg/util/handle_functions.hpp"
#include "dca/linalg/lapack/use_device.hpp"
#include "dca/linalg/blas/cublas1.hpp"
#include "dca/linalg/blas/cublas3.hpp"
#include "dca/linalg/blas/cublas_conversion_char_types.hpp"
#include "dca/linalg/blas/kernels_gpu.hpp"
//#include "dca/linalg/blas/blas3.hpp"
#include "dca/linalg/util/util_cublas.hpp"
//#include "dca/linalg/blas/use_device.hpp"
//#include "dca/linalg/blas/cublas3.hpp"
//#include "dca/linalg/linalg.hpp"
#include "dca/linalg/device_type.hpp"

namespace dca {
namespace phys {
namespace solver {
namespace ctaux {
// dca::phys::solver::accumulator::details::

using namespace linalg;
using linalg::util::castCudaComplex;
using linalg::util::CudaComplex;

std::array<int, 2> getBlockSize1D(const int ni, const int block_size) {
  const int n_threads = std::min(block_size, ni);
  const int n_blocks = dca::util::ceilDiv(ni, n_threads);
  return std::array<int, 2>{n_blocks, n_threads};
}


std::array<dim3, 2> getBlockSize(const uint i, const uint j, const uint block_size = 32) {
  const uint n_threads_i = std::min(block_size, i);
  const uint n_threads_j = std::min(block_size, j);
  if (n_threads_i * n_threads_j > 32 * 32)
    throw(std::logic_error("Block size is too big"));

  const uint n_blocks_i = dca::util::ceilDiv(i, n_threads_i);
  const uint n_blocks_j = dca::util::ceilDiv(j, n_threads_j);

  return std::array<dim3, 2>{dim3(n_blocks_i, n_blocks_j), dim3(n_threads_i, n_threads_j)};
}

std::array<dim3, 2> getBlockSize3D(const uint i, const uint j, const uint k) {
  const uint n_threads_k = std::min(uint(8), k);
  const uint max_block_size_ij = n_threads_k > 1 ? 8 : 32;
  const uint n_threads_i = std::min(max_block_size_ij, i);
  const uint n_threads_j = std::min(max_block_size_ij, j);

  const uint n_blocks_i = dca::util::ceilDiv(i, n_threads_i);
  const uint n_blocks_j = dca::util::ceilDiv(j, n_threads_j);
  const uint n_blocks_k = dca::util::ceilDiv(k, n_threads_k);

  return std::array<dim3, 2>{dim3(n_blocks_i, n_blocks_j, n_blocks_k),
                             dim3(n_threads_i, n_threads_j, n_threads_k)};
}



template <typename ScalarType>
__global__ void compute_G_r_t_up_Kernel(float* G0_M_G0_matrix_dev,int ldG0MG0, float*G_r_t,int ldGrt, int Gdmnsize , float* G0_matrix_left_dev, int ldG0_left, int config_size)
{

  const int n_rows = Gdmnsize;
  const int n_cols = Gdmnsize;


  const int id_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int id_j = blockIdx.y * blockDim.y + threadIdx.y;

  if (id_i >= n_rows || id_j >= n_cols)
    return;


  G_r_t[id_i + id_j*ldGrt] = tpeqtime_helper.G0_sign_up_mat(id_i,id_j) * (tpeqtime_helper.G0_original_up_mat(id_i,id_j) - G0_M_G0_matrix_dev[id_i + id_j*ldG0MG0]);

}

template <typename ScalarType>
__global__ void compute_G_r_t_dn_Kernel(float* G0_M_G0_matrix_dev, int ldG0MG0, float*G_r_t,int ldGrt, int Gdmnsize, float* G0_matrix_right_dev, int ldG0_right, int config_size )
{

  const int n_rows = Gdmnsize;
  const int n_cols = Gdmnsize;


  const int id_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int id_j = blockIdx.y * blockDim.y + threadIdx.y;

  if (id_i >= n_rows || id_j >= n_cols)
    return;


  G_r_t[id_i + id_j*ldGrt] = tpeqtime_helper.G0_sign_dn_mat(id_i,id_j) *( tpeqtime_helper.G0_original_dn_mat(id_i,id_j) - G0_M_G0_matrix_dev[id_i + id_j*ldG0MG0]);

}


template <typename ScalarType>
__global__ void compute_G0_matrix_Kernel(int spin_index, const ScalarType* M, int ldM, float* M_temp,int ldM_temp, float* G0_matrix_left_dev, int ldG0_left, float* G0_matrix_right_dev, int ldG0_right,  const ConfigElemTpEqTime* config_, int config_size,int Gdmnsize)
{

  const int n_rows = config_size;
  const int n_cols = Gdmnsize;

  const int id_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int id_j = blockIdx.y * blockDim.y + threadIdx.y;
  if (id_i >= n_rows || id_j >= n_cols)
    return;
  

  int r_ind_right, r_ind_left, b_i, b_j, r_i, r_j;    //, s_i, s_j;
  ScalarType t_i, t_j, delta_tau_right, delta_tau_left;  //, scaled_tau, f_tau;

    b_j = tpeqtime_helper.fixed_config_b_ind(id_j);
    r_j = tpeqtime_helper.fixed_config_r_ind(id_j);
    t_j = tpeqtime_helper.fixed_config_t_val(id_j);

      b_i = config_[id_i].band;
      r_i = config_[id_i].rsite;
      t_i = config_[id_i].tau;


	r_ind_right = tpeqtime_helper.rMinus(r_j,r_i);
        delta_tau_right = t_i - t_j;

	r_ind_left = tpeqtime_helper.rMinus(r_i,r_j);
        delta_tau_left = t_j - t_i;

       G0_matrix_right_dev[id_i + id_j*ldG0_right] = tpeqtime_helper.akima_coeff_mat(b_i, spin_index, b_j, spin_index, r_ind_right, delta_tau_right); 
       G0_matrix_left_dev[id_j + id_i*ldG0_left] = tpeqtime_helper.akima_coeff_mat(b_j, spin_index, b_i, spin_index, r_ind_left, delta_tau_left); 
}



template <typename ScalarType>
__global__ void setM_temp_Kernel(const ScalarType * M, int ldM, float* M_temp, int ldM_temp, int config_size)
{

  const int n_rows = config_size;
  const int n_cols = config_size;
  
  const int id_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int id_j = blockIdx.y * blockDim.y + threadIdx.y;

  if (id_i >= n_rows || id_j >= n_cols)
    return;


  M_temp[id_i + id_j*ldM_temp] = float(M[id_i + id_j*ldM]); 

}

__global__ void gemm_Kernel(int n_sum,int n_rows, int n_cols,float * A, int lda, float* B, int ldb, float* C, int ldc)
{


  const int id_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int id_j = blockIdx.y * blockDim.y + threadIdx.y;

  if (id_i >= n_rows || id_j >= n_cols)
    return;


	float summed=0.0;
   for (int k=0; k<n_sum; k++)
	{
	summed += A[id_i + lda*k]*B[k + ldb*id_j];
	} 

	C[id_i+ldc*id_j] = summed;
}

template <typename ScalarType>
void calc_G_r_t_OnDevice(int spin_index, const ScalarType* M, int ldM, float* M_temp, int ldM_temp, float* G0_matrix_left_dev, int ldG0_left, float* G0_matrix_right_dev, int ldG0_right, float* M_G0_matrix_dev, int ldMG0, float* G0_M_G0_matrix_dev, int ldG0MG0, const ConfigElemTpEqTime* config_, int config_size, float* G_r_t, int ldGrt, int Gdmnsize, cudaStream_t stream_, int stream_id, int thread_id)
{
  const int n_rows = config_size;
  const int n_cols = Gdmnsize;

  auto blocks_right = getBlockSize(n_rows, n_rows);
  auto blocks_left = getBlockSize(n_cols, n_cols);
  auto blocks = getBlockSize(n_rows, n_cols);

     compute_G0_matrix_Kernel<ScalarType><<<blocks[0], blocks[1], 0, stream_>>>(spin_index, M, ldM, M_temp, ldM_temp, G0_matrix_left_dev, ldG0_left, G0_matrix_right_dev,ldG0_right, config_, config_size, Gdmnsize);

     setM_temp_Kernel<ScalarType><<<blocks_right[0], blocks_right[1], 0, stream_>>>(M, ldM, M_temp, ldM_temp, config_size);

/*
    cudaStreamSynchronize(stream_);

	float alpha=1.0;
	float beta=0.0;
	cublasHandle_t handle0 = dca::linalg::util::getHandle(thread_id, stream_id);
    dca::linalg::cublas::gemm(handle0,"N", "N", n_rows, n_cols, n_rows , alpha, M_temp, ldM_temp, G0_matrix_right_dev, ldG0_right, beta, M_G0_matrix_dev, ldMG0);
    dca::linalg::cublas::gemm(handle0,"N", "N", n_cols, n_cols, n_rows , alpha, G0_matrix_left_dev, ldG0_left, M_G0_matrix_dev, ldMG0, beta, G0_M_G0_matrix_dev, ldG0MG0);


    cudaStreamSynchronize(stream_);

*/

    gemm_Kernel<<<blocks[0], blocks[1], 0, stream_>>>(n_rows,n_rows, n_cols, M_temp, ldM_temp, G0_matrix_right_dev, ldG0_right, M_G0_matrix_dev, ldMG0);
    gemm_Kernel<<<blocks_left[0], blocks_left[1], 0, stream_>>>(n_rows,n_cols,n_cols,G0_matrix_left_dev, ldG0_left, M_G0_matrix_dev, ldMG0, G0_M_G0_matrix_dev, ldG0MG0);

    if (spin_index==1)
     compute_G_r_t_up_Kernel<ScalarType><<<blocks_left[0], blocks_left[1], 0, stream_>>>(G0_M_G0_matrix_dev, ldG0MG0, G_r_t, ldGrt, Gdmnsize, G0_matrix_left_dev, ldG0_left, config_size);
    else     
     compute_G_r_t_dn_Kernel<ScalarType><<<blocks_left[0], blocks_left[1], 0, stream_>>>(G0_M_G0_matrix_dev, ldG0MG0, G_r_t, ldGrt, Gdmnsize, G0_matrix_right_dev,ldG0_right,config_size);

}

template <typename ScalarType>
__global__ void accumulate_G_r_t_OnDevice_Kernel(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, ScalarType sign, double* G_r_t_accumulated, double* G_r_t_accumulated_squared)
{

  const int n_rows = tpeqtime_helper.get_b_r_t_VERTEX_dmn_tsize();
  const int n_cols = tpeqtime_helper.get_b_r_t_VERTEX_dmn_tsize();


  const int id_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int id_j = blockIdx.y * blockDim.y + threadIdx.y;

  double dn_contribution, up_contribution, dn_contribution_2, up_contribution_2;


  if (id_i >= n_rows || id_j >= n_cols)
    return;

  int index_up  = tpeqtime_helper.G0_indices_up_mat(id_i,id_j);
  int index_dn  = tpeqtime_helper.G0_indices_dn_mat(id_i,id_j);

  dn_contribution = sign*double(tpeqtime_helper.G0_integration_factor_dn_mat(id_i,id_j) * G_r_t_dn[id_i + id_j*ldGrt_dn]);
  up_contribution = sign*double(tpeqtime_helper.G0_integration_factor_up_mat(id_i,id_j) * G_r_t_up[id_i + id_j*ldGrt_up]);

  dn_contribution_2 = sign*double(tpeqtime_helper.G0_integration_factor_dn_mat(id_i,id_j) * G_r_t_dn[id_i + id_j*ldGrt_dn]*G_r_t_dn[id_i + id_j*ldGrt_dn]);
  up_contribution_2 = sign*double(tpeqtime_helper.G0_integration_factor_up_mat(id_i,id_j) * G_r_t_up[id_i + id_j*ldGrt_up]*G_r_t_up[id_i + id_j*ldGrt_up]);

  atomicAdd(&G_r_t_accumulated[index_dn], dn_contribution);
  atomicAdd(&G_r_t_accumulated[index_up], up_contribution);

  atomicAdd(&G_r_t_accumulated_squared[index_dn], dn_contribution_2);
  atomicAdd(&G_r_t_accumulated_squared[index_up], up_contribution_2);



}

template <typename ScalarType>
void accumulate_G_r_t_OnDevice(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, ScalarType sign, double* G_r_t_accumulated, double* G_r_t_accumulated_squared, int G0dmnsize, cudaStream_t stream_)
{

  const int n_rows = G0dmnsize;
  const int n_cols = G0dmnsize;
  auto blocks = getBlockSize(n_rows, n_cols);

     accumulate_G_r_t_OnDevice_Kernel<ScalarType><<<blocks[0], blocks[1], 0, stream_>>>(G_r_t_up, ldGrt_up, G_r_t_dn, ldGrt_dn, sign, G_r_t_accumulated, G_r_t_accumulated_squared);

}


template <typename ScalarType>
__global__ void accumulate_chi_OnDevice_Kernel(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, ScalarType sign, double* spin_ZZ_chi_accumulated, double* spin_ZZ_chi_stddev, double* spin_XX_chi_accumulated, double sfactor,int t_VERTEX_dmn_size)
{

  const int n_rows = tpeqtime_helper.get_b_r_t_VERTEX_dmn_tsize();
  const int n_cols = tpeqtime_helper.get_b_r_t_VERTEX_dmn_tsize();


  const int id_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int id_j = blockIdx.y * blockDim.y + threadIdx.y;
   
  if (id_i >= n_rows || id_j >= n_cols)
    return;

  int b_i, b_j, r_i, r_j, t_i, t_j, dr_index, dt, index, indexL;
  float upup, updn, spin_ZZ_val, spin_XX_contribution;
  double dr;

 
    b_j = tpeqtime_helper.fixed_config_b_ind(id_j);
    r_j = tpeqtime_helper.fixed_config_r_ind(id_j);
    t_j = tpeqtime_helper.fixed_config_t_ind(id_j);


      b_i = tpeqtime_helper.fixed_config_b_ind(id_i);
      r_i = tpeqtime_helper.fixed_config_r_ind(id_i);
      t_i = tpeqtime_helper.fixed_config_t_ind(id_i);


      dr_index = tpeqtime_helper.rMinus(r_j,r_i);
      dr = tpeqtime_helper.r_abs_diff(dr_index);

      dt = t_i-t_j;

      spin_ZZ_val = 0.0;
      spin_XX_contribution = 0.0;

      if(t_i != t_VERTEX_dmn_size-1 && t_j != t_VERTEX_dmn_size-1)
      {
        dt = dt<0 ? dt+t_VERTEX_dmn_size-1 : dt;

	index = tpeqtime_helper.chi_index(b_i,b_j,dr_index,dt);

        upup = G_r_t_up[id_i+id_j*ldGrt_up]*G_r_t_up[id_j+id_i*ldGrt_up] + G_r_t_dn[id_i+id_j*ldGrt_dn]*G_r_t_dn[id_j+id_i*ldGrt_dn];
        updn = G_r_t_dn[id_i+id_j*ldGrt_dn]*G_r_t_up[id_j+id_i*ldGrt_up] + G_r_t_up[id_i+id_j*ldGrt_up]*G_r_t_dn[id_j+id_i*ldGrt_dn];


        if(dt==0){
          spin_XX_contribution -= sfactor* updn*sign;
          spin_ZZ_val = -upup;
        } else{
          spin_XX_contribution += sfactor* updn*sign;
          spin_ZZ_val = upup;
        }

        upup = (1.0+G_r_t_up[id_i+id_i*ldGrt_up])*(1.0+G_r_t_up[id_j+id_j*ldGrt_up]) + (1.0+G_r_t_dn[id_i+id_i*ldGrt_dn])*(1.0+G_r_t_dn[id_j+id_j*ldGrt_dn]);
        updn = (1.0+G_r_t_up[id_i+id_i*ldGrt_up])*(1.0+G_r_t_dn[id_j+id_j*ldGrt_dn]) + (1.0+G_r_t_dn[id_i+id_i*ldGrt_dn])*(1.0+G_r_t_up[id_j+id_j*ldGrt_up]);
        spin_ZZ_val += (upup - updn);

        if(b_i==b_j && dr<5e-7 && dt==0){
          // correction due to cc+ = 1=c+c

          updn = G_r_t_up[id_j+id_j*ldGrt_up] + G_r_t_dn[id_j+id_j*ldGrt_dn];
          spin_XX_contribution -= sfactor* updn*sign;
          spin_ZZ_val += -updn;
        }
        atomicAdd(&spin_XX_chi_accumulated[index],double(spin_XX_contribution));
        atomicAdd(&spin_ZZ_chi_accumulated[index], double(spin_ZZ_val * sfactor * sign));
        atomicAdd(&spin_ZZ_chi_stddev[index], double(spin_ZZ_val * spin_ZZ_val * sfactor * sign));

      if(dt==0)
	{
        indexL = tpeqtime_helper.chi_index(b_i,b_j,dr_index,t_VERTEX_dmn_size-1);
        atomicAdd(&spin_XX_chi_accumulated[indexL],double(spin_XX_contribution));
        atomicAdd(&spin_ZZ_chi_accumulated[indexL], double(spin_ZZ_val * sfactor * sign));
        atomicAdd(&spin_ZZ_chi_stddev[indexL], double(spin_ZZ_val * spin_ZZ_val * sfactor * sign));

	}

      }


}



template <typename ScalarType>
void accumulate_chi_OnDevice(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, ScalarType sign, double* spin_ZZ_chi_accumulated, double* spin_ZZ_chi_stddev, double* spin_XX_chi_accumulated, int G0dmnsize, int r_dmn_t_dmn_size ,int t_VERTEX_dmn_size, cudaStream_t stream_)
{

  const int n_rows = G0dmnsize;
  const int n_cols = G0dmnsize;
  auto blocks = getBlockSize(n_rows, n_cols,24);
  //auto blocks = getBlockSize(n_rows, n_cols); //not used due to too many registers

  double sfactor = 0.5/double(((t_VERTEX_dmn_size-1.0)*r_dmn_t_dmn_size));

    accumulate_chi_OnDevice_Kernel<<<blocks[0], blocks[1], 0, stream_>>>(G_r_t_up, ldGrt_up, G_r_t_dn, ldGrt_dn, sign, spin_ZZ_chi_accumulated, spin_ZZ_chi_stddev, spin_XX_chi_accumulated, sfactor,t_VERTEX_dmn_size);

}




__global__ void sum_OnDevice_Kernel( double* inMatrix, double* outMatrix, int ldM)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < ldM) {
    //outMatrix[i] += inMatrix[i];
    atomicAdd(&outMatrix[i],inMatrix[i]);
  }
}



void sum_OnDevice(double* inMatrix, double* outMatrix, int ldM, cudaStream_t stream_)
{

  const int n = ldM;
  auto blocks = getBlockSize1D(n,128);

     sum_OnDevice_Kernel<<<blocks[0], blocks[1], 0, stream_>>>(inMatrix, outMatrix, ldM);

}


template void accumulate_chi_OnDevice<double>(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, double sign, double* spin_ZZ_chi_accumulated, double* spin_ZZ_stddev, double* spin_XX_chi_accumulated, int G0dmnsize, int r_dmn_t_dmn_size ,int t_VERTEX_dmn_size, cudaStream_t stream_);


template void accumulate_chi_OnDevice<float>(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, float sign, double* spin_ZZ_chi_accumulated, double* spin_ZZ_stddev, double* spin_XX_chi_accumulated, int G0dmnsize, int r_dmn_t_dmn_size ,int t_VERTEX_dmn_size, cudaStream_t stream_);

template void accumulate_G_r_t_OnDevice<double>(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, double sign, double* G_r_t_accumulated, double* G_r_t_accumulated_squared, int G0dmnsize, cudaStream_t stream_);

template void accumulate_G_r_t_OnDevice<float>(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, float sign, double* G_r_t_accumulated, double* G_r_t_accumulated_squared, int G0dmnsize, cudaStream_t stream_);

template void calc_G_r_t_OnDevice<double>(int spin_index, const double* M, int ldM, float* M_temp, int ldM_temp, float* G0_matrix_left_dev, int ldG0_left, float* G0_matrix_right_dev, int ldG0_right, float* M_G0_matrix_dev, int ldMG0, float* G0_M_G0_matrix_dev, int ldG0MG0, const ConfigElemTpEqTime* config_, int config_size, float* G_r_t, int ldGrt, int Gdmnsize, cudaStream_t stream_, int stream_id, int thread_id);


template void calc_G_r_t_OnDevice<float>(int spin_index, const float* M, int ldM, float* M_temp, int ldM_temp, float* G0_matrix_left_dev, int ldG0_left, float* G0_matrix_right_dev, int ldG0_right, float* M_G0_matrix_dev, int ldMG0, float* G0_M_G0_matrix_dev, int ldG0MG0, const ConfigElemTpEqTime* config_, int config_size, float* G_r_t, int ldGrt, int Gdmnsize, cudaStream_t stream_, int stream_id, int thread_id);

}  // namespace ctaux
}  // namespace solver
}  // namespace phys
}  // namespace dca
