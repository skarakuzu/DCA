// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Helper class for adding and subtracting momentum and frequency on the device.

#ifndef DCA_INCLUDE_DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTAUX_ACCUMULATOR_TP_TPEQTIME_HELPER_CUH
#define DCA_INCLUDE_DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTAUX_ACCUMULATOR_TP_TPEQTIME_HELPER_CUH


#include <vector>

#include <cuda.h>

namespace dca {
namespace phys {
namespace solver {
namespace ctaux {
// dca::phys::solver::ctaux::

class TpEqTimeHelper {
public:

  static void set(const int* sub_r, int lds, int nr, const int* G0_indices_up, int ldG0_indices_up, const int* G0_indices_dn, int ldG0_indices_dn, const float* G0_sign_up, int ldG0_sign_up, const float* G0_sign_dn, int ldG0_sign_dn, const float* G0_integration_factor_up, int ldG0_integration_factor_up, const float* G0_integration_factor_dn,int  ldG0_integration_factor_dn, const float* G0_original_up, int ldG0_original_up,  const float* G0_original_dn, int ldG0_original_dn, int b_r_t_VERTEX_dmn_t_size,int t_VERTEX_dmn_size, const double* akima_coeff, int lakm, int nb_akm, int ns_akm, int nr_akm, int nt_akm, int akima_size, int* fixed_config_b_ind, int* fixed_config_r_ind, int* fixed_config_t_ind, double* fixed_config_t_val,  double beta, double N_div_beta);


  __device__ inline int get_b_r_t_VERTEX_dmn_tsize() const;
  __device__ inline int get_t_VERTEX_dmn_size() const;
  __device__ inline int get_r_dmn_t_size() const;
  __device__ inline int rMinus(int r_idx, int r_idj) const;
  __device__ inline float G0_sign_up_mat(int r_idx, int r_idj) const;
  __device__ inline float G0_sign_dn_mat(int r_idx, int r_idj) const;
  __device__ inline int G0_indices_up_mat(int r_idx, int r_idj) const;
  __device__ inline int G0_indices_dn_mat(int r_idx, int r_idj) const;
  __device__ inline float G0_integration_factor_up_mat(int r_idx, int r_idj) const;
  __device__ inline float G0_integration_factor_dn_mat(int r_idx, int r_idj) const;
  __device__ inline float G0_original_up_mat(int r_idx, int r_idj) const;
  __device__ inline float G0_original_dn_mat(int r_idx, int r_idj) const;
  __device__ inline int fixed_config_b_ind(int index) const;
  __device__ inline int fixed_config_r_ind(int index) const;
  __device__ inline int fixed_config_t_ind(int index) const;
  __device__ inline double fixed_config_t_val(int index) const;
  __device__ inline int chi_index(int b1, int b2, int dr, int dt) const;
  __device__ inline double akima_coeff_mat(int b1, int s1, int b2, int s2, int r_ind, double delta_tau) const;

protected:
  int lda_;
  int lds_;
  int bVrtxdmn_;
  int tVrtxdmn_;
  int rDmnt_;
  int ldG0_indices_up_;
  int ldG0_indices_dn_;
  int ldG0_sign_up_;
  int ldG0_sign_dn_;
  int ldG0_integration_factor_up_;
  int ldG0_integration_factor_dn_;
  int ldG0_original_up_;
  int ldG0_original_dn_;
  int size_config_;
  double beta_;
  double N_div_beta_;
  unsigned akima_steps_[7];
  unsigned chi_steps_[4];
  int* sub_matrix_;
  int* G0_indices_up_;
  int* G0_indices_dn_;
  double* akima_coefficients_;
  float* G0_sign_up_;
  float* G0_sign_dn_;
  float* G0_integration_factor_up_;
  float* G0_integration_factor_dn_;
  float* G0_original_up_;
  float* G0_original_dn_;
  int* fixed_config_b_ind_;
  int* fixed_config_r_ind_;
  int* fixed_config_t_ind_;
  double* fixed_config_t_val_;
};

// Global instance to be used in the tp accumulation kernel.
extern __device__ __constant__ TpEqTimeHelper tpeqtime_helper;

inline __device__ int TpEqTimeHelper::get_b_r_t_VERTEX_dmn_tsize() const {
  return bVrtxdmn_;
}
inline __device__ int TpEqTimeHelper::get_t_VERTEX_dmn_size() const {
  return tVrtxdmn_;
}
inline __device__ int TpEqTimeHelper::get_r_dmn_t_size() const {
  return rDmnt_;
}

inline __device__ int TpEqTimeHelper::rMinus(const int r_idx, const int r_idj) const {
  return sub_matrix_[r_idx + lds_ * r_idj];
}

inline __device__ float TpEqTimeHelper::G0_sign_up_mat(const int idx, const int idj) const {
  return G0_sign_up_[idx + ldG0_sign_up_ * idj];
}

inline __device__ float TpEqTimeHelper::G0_sign_dn_mat(const int idx, const int idj) const {
  return G0_sign_dn_[idx + ldG0_sign_dn_ * idj];
}

inline __device__ int TpEqTimeHelper::G0_indices_up_mat(const int idx, const int idj) const {
  return G0_indices_up_[idx + ldG0_indices_up_ * idj];
}

inline __device__ int TpEqTimeHelper::G0_indices_dn_mat(const int idx, const int idj) const {
  return G0_indices_dn_[idx + ldG0_indices_dn_ * idj];
}

inline __device__ float TpEqTimeHelper::G0_integration_factor_up_mat(const int idx, const int idj) const {
  return G0_integration_factor_up_[idx + ldG0_integration_factor_up_ * idj];
}

inline __device__ float TpEqTimeHelper::G0_integration_factor_dn_mat(const int idx, const int idj) const {
  return G0_integration_factor_dn_[idx + ldG0_integration_factor_dn_ * idj];
}

inline __device__ float TpEqTimeHelper::G0_original_up_mat(const int idx, const int idj) const {
  return G0_original_up_[idx + ldG0_original_up_ * idj];
}

inline __device__ float TpEqTimeHelper::G0_original_dn_mat(const int idx, const int idj) const {
  return G0_original_dn_[idx + ldG0_original_dn_ * idj];
}


inline __device__ int TpEqTimeHelper::fixed_config_b_ind(const int index) const {
  return fixed_config_b_ind_[index];
}

inline __device__ int TpEqTimeHelper::fixed_config_r_ind(const int index) const {
  return fixed_config_r_ind_[index];
}

inline __device__ int TpEqTimeHelper::fixed_config_t_ind(const int index) const {
  return fixed_config_t_ind_[index];
}

inline __device__ double TpEqTimeHelper::fixed_config_t_val(const int index) const {
  return fixed_config_t_val_[index];
}

inline __device__ int TpEqTimeHelper::chi_index(const int b1, const int b2, const int dr, const int dt) const {
  int index = chi_steps_[0]*b1 + chi_steps_[1]*b2 + chi_steps_[2]*dr + chi_steps_[3]*dt;
  return index;
}

inline __device__ double TpEqTimeHelper::akima_coeff_mat(const int b1, const int s1, const int b2, const int s2, const int r_ind, const double tau) const {

  double new_tau = tau + beta_;

  double scaled_tau = new_tau * N_div_beta_;

  int t_ind = scaled_tau;

  double delta_tau = scaled_tau - t_ind;

  int linind =  akima_steps_[0]*(akima_steps_[1] * b1 + akima_steps_[2] * s1 + akima_steps_[3] * b2 + 
         akima_steps_[4] * s2 + akima_steps_[5] * r_ind + akima_steps_[6] * t_ind);  	


  double* a_ptr = &akima_coefficients_[linind];

  double result =
      (a_ptr[0] + delta_tau * (a_ptr[1] + delta_tau * (a_ptr[2] + delta_tau * a_ptr[3])));

  //double result = b1+s1+b2+s2+r_ind+t_ind;

  return result;

}

}  // namespace ctaux
}  // namespace solver
}  // namespace phys
}  // namespace dca

#endif  // DCA_INCLUDE_DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTAUX_ACCUMULATOR_TPEQTIME_HELPER_CUH
