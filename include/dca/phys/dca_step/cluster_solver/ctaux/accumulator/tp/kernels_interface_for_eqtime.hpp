// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@gitp.phys.ethz.ch)
//
// This file provides the interface to the kernels for the Dnfft1DGpu accumulator

#ifndef DCA_INCLUDE_DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTAUX_ACCUMULATOR_TP_KERNELS_INTERFACE_FOR_EQTIME_HPP
#define DCA_INCLUDE_DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTAUX_ACCUMULATOR_TP_KERNELS_INTERFACE_FOR_EQTIME_HPP

#ifndef DCA_HAVE_CUDA
#pragma error "This file requires CUDA."
#endif

//#include "dca/phys/dca_step/cluster_solver/ctaux/accumulator/tp/singleton_obj_dev.hpp"

#include <cuda.h>

namespace dca {
namespace phys {
namespace solver {
namespace ctaux {
// dca::phys::solver::ctaux::

struct ConfigElemTpEqTime {
  int band;
  int rsite;
  double tau;
};

template <typename ScalarType>
void calc_G_r_t_OnDevice(int spin_index, const ScalarType* M, int ldM, float* M_temp, int ldM_temp, float* G0_matrix_left_dev, int ldG0_left, float* G0_matrix_right_dev, int ldG0_right, float* M_G0_matrix_dev, int ldMG0, float* G0_M_G0_matrix_dev, int ldG0MG0, const ConfigElemTpEqTime* config_, int config_size, float* G_r_t, int ldGrt, int Gdmnsize, cudaStream_t stream_, int stream_id, int thread_id);

template <typename ScalarType>
void accumulate_G_r_t_OnDevice(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, ScalarType sign, double* G_r_t_accumulated, double* G_r_t_accumulated_squared, int Gdmnsize, cudaStream_t stream_);


template <typename ScalarType>
void accumulate_chi_OnDevice(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, ScalarType sign, double* spin_ZZ_chi_accumulated, double* spin_ZZ_stddev, double* spin_XX_chi_accumulated, int G0dmnsize, int r_dmn_t_dmn_size ,int t_VERTEX_dmn_size, cudaStream_t stream_);

template <typename ScalarType>
void accumulate_dwave_pp_correlator_OnDevice(const float * G_r_t_up, int ldGrt_up, const float* G_r_t_dn, int ldGrt_dn, ScalarType sign, double* dwave_pp_correlator, int b_r_t_VERTEX_dmn_size, int dwave_config_size, cudaStream_t stream_);

void sum_OnDevice(double* inMatrix, double* outMatrix, int ldM, cudaStream_t stream_);

}  // ctaux
}  // solver
}  // phys
}  // dca

#endif  // DCA_MATH_NFFT_KERNELS_INTERFACE_HPP
