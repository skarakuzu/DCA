// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file implements G4Helper::set.

#include "dca/phys/dca_step/cluster_solver/ctaux/accumulator/tp/TpEqTime_helper.cuh"

#include <algorithm>
#include <array>
#include <mutex>
#include <stdexcept>

namespace dca {
namespace phys {
namespace solver {
namespace ctaux {
// dca::phys::solver::ctaux::

__device__ __constant__ TpEqTimeHelper tpeqtime_helper;

void TpEqTimeHelper::set(const int* sub_r, int lds, int nr_sub, const int* G0_indices_up, int ldG0_indices_up, const int* G0_indices_dn, int ldG0_indices_dn, const float* G0_sign_up, int ldG0_sign_up,  const float* G0_sign_dn, int ldG0_sign_dn, const float* G0_integration_factor_up, int ldG0_integration_factor_up, const float* G0_integration_factor_dn, int ldG0_integration_factor_dn, const float* G0_original_up, int ldG0_original_up, const float* G0_original_dn, int ldG0_original_dn,  int G0dmnsize, int tVertex_dmnsize,  const double* akima_coeff, int lakm, int nb_akm, int ns_akm, int nr_akm, int nt_akm, int akima_size, int* fixed_config_b_ind, int* fixed_config_r_ind, int* fixed_config_t_ind, double* fixed_config_t_val, double* r_abs_diff, int cluster_size,double beta, double N_div_beta, int* dwave_config_r_l, int* dwave_config_r_lp, int* dwave_config_b_l, int* dwave_config_b_lp, int dwave_config_size, double* dwave_r_factor) {


  static std::once_flag flag;

  std::call_once(flag, [=]() {
    TpEqTimeHelper host_helper;
    host_helper.bVrtxdmn_ = G0dmnsize;
    host_helper.tVrtxdmn_ = tVertex_dmnsize;
    host_helper.rDmnt_ = nr_akm;
    host_helper.dwave_config_size_ = dwave_config_size;
    host_helper.lds_ = lds;
    host_helper.ldG0_indices_up_ = ldG0_indices_up;
    host_helper.ldG0_indices_dn_ = ldG0_indices_dn;
    host_helper.ldG0_sign_up_ = ldG0_sign_up;
    host_helper.ldG0_sign_dn_ = ldG0_sign_dn;
    host_helper.ldG0_integration_factor_up_ = ldG0_integration_factor_up;
    host_helper.ldG0_integration_factor_dn_ = ldG0_integration_factor_dn;
    host_helper.ldG0_original_up_ = ldG0_original_up;
    host_helper.ldG0_original_dn_ = ldG0_original_dn;
    host_helper.cluster_size_ = cluster_size;
    host_helper.N_div_beta_ = N_div_beta;
    host_helper.beta_ = beta;

/*
    host_helper.ext_size_ = 0;
    for (const int idx : delta_w)
      host_helper.ext_size_ = std::max(host_helper.ext_size_, std::abs(idx));
*/

    const std::array<int, 7> akima_sizes{lakm,
				    nb_akm,
                                    ns_akm,
                                    nb_akm,
                                    ns_akm,
                                    nr_akm,
                                    nt_akm};

    std::array<int, 7> steps;
    steps[0] = 1;
    for (std::size_t i = 1; i < steps.size(); ++i)
      steps[i] = steps[i - 1] * akima_sizes[i - 1];

    std::copy_n(steps.data(), steps.size(), host_helper.akima_steps_);



    const std::array<int, 4> chi_sizes{nb_akm,
				  nb_akm,
				  nr_akm,	
				  tVertex_dmnsize};

    std::array<int, 4> steps2;
    steps2[0] = 1;
    for (std::size_t i = 1; i < steps2.size(); ++i)
      steps2[i] = steps2[i - 1] * chi_sizes[i - 1];


    std::copy_n(steps2.data(), steps2.size(), host_helper.chi_steps_);

    const std::array<int, 3> brt_dmn_sizes{nb_akm,
				  nr_akm,	
				  tVertex_dmnsize};

    std::array<int, 3> steps3;
    steps3[0] = 1;
    for (std::size_t i = 1; i < steps3.size(); ++i)
      steps3[i] = steps3[i - 1] * brt_dmn_sizes[i - 1];


    std::copy_n(steps3.data(), steps3.size(), host_helper.brt_dmn_steps_);

    cudaMalloc(&host_helper.sub_matrix_, sizeof(int) * lds * nr_sub);
    cudaMemcpy(host_helper.sub_matrix_, sub_r, sizeof(int) * lds * nr_sub, cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.G0_indices_up_, sizeof(int) * ldG0_indices_up * G0dmnsize);
    cudaMemcpy(host_helper.G0_indices_up_, G0_indices_up, sizeof(int) * ldG0_indices_up * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.G0_indices_dn_, sizeof(int) * ldG0_indices_dn * G0dmnsize);
    cudaMemcpy(host_helper.G0_indices_dn_, G0_indices_dn, sizeof(int) * ldG0_indices_dn * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.G0_sign_up_, sizeof(float) * ldG0_sign_up * G0dmnsize);
    cudaMemcpy(host_helper.G0_sign_up_, G0_sign_up, sizeof(float) * ldG0_sign_up * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.G0_sign_dn_, sizeof(float) * ldG0_sign_dn * G0dmnsize);
    cudaMemcpy(host_helper.G0_sign_dn_, G0_sign_dn, sizeof(float) * ldG0_sign_dn * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.G0_integration_factor_up_, sizeof(float) * ldG0_integration_factor_up * G0dmnsize);
    cudaMemcpy(host_helper.G0_integration_factor_up_, G0_integration_factor_up, sizeof(float) * ldG0_integration_factor_up * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.G0_integration_factor_dn_, sizeof(float) * ldG0_integration_factor_dn * G0dmnsize);
    cudaMemcpy(host_helper.G0_integration_factor_dn_, G0_integration_factor_dn, sizeof(float) * ldG0_integration_factor_dn * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.G0_original_up_, sizeof(float) * ldG0_original_up * G0dmnsize);
    cudaMemcpy(host_helper.G0_original_up_, G0_original_up, sizeof(float) * ldG0_original_up *G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.G0_original_dn_, sizeof(float) * ldG0_original_dn *G0dmnsize);
    cudaMemcpy(host_helper.G0_original_dn_, G0_original_dn, sizeof(float) * ldG0_original_dn *G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.akima_coefficients_, sizeof(double) * akima_size);
    cudaMemcpy(host_helper.akima_coefficients_, akima_coeff, sizeof(double) * akima_size,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.fixed_config_b_ind_, sizeof(int) * G0dmnsize);
    cudaMemcpy(host_helper.fixed_config_b_ind_, fixed_config_b_ind, sizeof(int) * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.fixed_config_r_ind_, sizeof(int) * G0dmnsize);
    cudaMemcpy(host_helper.fixed_config_r_ind_, fixed_config_r_ind, sizeof(int) * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.fixed_config_t_ind_, sizeof(int) *G0dmnsize);
    cudaMemcpy(host_helper.fixed_config_t_ind_, fixed_config_t_ind, sizeof(int) * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.fixed_config_t_val_, sizeof(double) * G0dmnsize);
    cudaMemcpy(host_helper.fixed_config_t_val_, fixed_config_t_val, sizeof(double) * G0dmnsize,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.r_abs_diff_, sizeof(double) * cluster_size);
    cudaMemcpy(host_helper.r_abs_diff_, r_abs_diff, sizeof(double) * cluster_size,
               cudaMemcpyHostToDevice);


    cudaMalloc(&host_helper.dwave_config_r_l_, sizeof(int) * dwave_config_size);
    cudaMemcpy(host_helper.dwave_config_r_l_, dwave_config_r_l, sizeof(int) * dwave_config_size,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.dwave_config_r_lp_, sizeof(int) * dwave_config_size);
    cudaMemcpy(host_helper.dwave_config_r_lp_, dwave_config_r_lp, sizeof(int) * dwave_config_size,
               cudaMemcpyHostToDevice);


    cudaMalloc(&host_helper.dwave_config_b_l_, sizeof(int) * dwave_config_size);
    cudaMemcpy(host_helper.dwave_config_b_l_, dwave_config_b_l, sizeof(int) * dwave_config_size,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.dwave_config_b_lp_, sizeof(int) * dwave_config_size);
    cudaMemcpy(host_helper.dwave_config_b_lp_, dwave_config_b_lp, sizeof(int) * dwave_config_size,
               cudaMemcpyHostToDevice);

    cudaMalloc(&host_helper.dwave_r_factor_, sizeof(double) * nr_akm);
    cudaMemcpy(host_helper.dwave_r_factor_, dwave_r_factor, sizeof(double) * nr_akm,
               cudaMemcpyHostToDevice);


    cudaMemcpyToSymbol(tpeqtime_helper, &host_helper, sizeof(TpEqTimeHelper));
  });
}

}  // namespace ctaux
}  // namespace solver
}  // namespace phys
}  // namespace dca
