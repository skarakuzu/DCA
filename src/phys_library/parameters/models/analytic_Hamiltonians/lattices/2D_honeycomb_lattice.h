// Copyright (C) 2009-2016 ETH Zurich
// Copyright (C) 2007?-2016 Center for Nanophase Materials Sciences, ORNL
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Peter Staar (peter.w.j.staar@gmail.com)
//
// \f{eqnarray}{
//   H_{i,j}(\vec{k}) &=& \delta_{\sigma_i,\sigma_j}...
// \f}

#ifndef PHYS_LIBRARY_PARAMETERS_MODELS_ANALYTIC_HAMILTONIANS_LATTICES_2D_HONEYCOMB_LATTICE_H
#define PHYS_LIBRARY_PARAMETERS_MODELS_ANALYTIC_HAMILTONIANS_LATTICES_2D_HONEYCOMB_LATTICE_H

#include <cmath>
#include <complex>
#include <utility>
#include <vector>

#include "enumerations.hpp"
#include "comp_library/function_library/include_function_library.h"
#include "phys_library/domains/cluster/symmetries/point_groups/No_symmetry.h"

template <typename DCA_point_group_type>
class graphene_model {
public:
  typedef no_symmetry<2> LDA_point_group;
  typedef DCA_point_group_type DCA_point_group;

  const static cluster_shape_type DCA_cluster_shape = BETT_CLUSTER;
  const static cluster_shape_type LDA_cluster_shape = PARALLELEPIPED;

  const static int DIMENSION = 2;
  const static int BANDS = 2;

  static double* initialize_r_DCA_basis();
  static double* initialize_k_DCA_basis();

  static double* initialize_r_LDA_basis();
  static double* initialize_k_LDA_basis();

  static std::vector<int> get_flavors();
  static std::vector<std::vector<double>> get_a_vectors();

  std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> get_orbital_permutations();

  template <class domain, class parameters_type>
  static void initialize_H_interaction(FUNC_LIB::function<double, domain>& H_interaction,
                                       parameters_type& parameters);

  template <class domain>
  static void initialize_H_symmetry(FUNC_LIB::function<int, domain>& H_symmetry);

  template <class parameters_type>
  static std::complex<double> get_LDA_Hamiltonians(parameters_type& parameters, std::vector<double> k,
                                                   int b1, int s1, int b2, int s2);
};

template <typename DCA_point_group_type>
double* graphene_model<DCA_point_group_type>::initialize_r_DCA_basis() {
  static double* r_DCA = new double[4];

  r_DCA[0] = 1.5;
  r_DCA[1] = 0.866025403784;
  r_DCA[2] = 1.5;
  r_DCA[3] = -0.866025403784;

  return r_DCA;
}

template <typename DCA_point_group_type>
double* graphene_model<DCA_point_group_type>::initialize_k_DCA_basis() {
  static double* k_DCA = new double[4];

  k_DCA[0] = 2.09439466667;
  k_DCA[1] = 3.62759797377;
  k_DCA[2] = 2.09439466667;
  k_DCA[3] = -3.62759797377;

  return k_DCA;
}

template <typename DCA_point_group_type>
double* graphene_model<DCA_point_group_type>::initialize_r_LDA_basis() {
  static double* r_LDA = new double[4];

  r_LDA[0] = 1. / 2.;
  r_LDA[1] = 0;
  r_LDA[2] = 0.;
  r_LDA[3] = std::sqrt(3.) / 2.;

  return r_LDA;
}

template <typename DCA_point_group_type>
double* graphene_model<DCA_point_group_type>::initialize_k_LDA_basis() {
  static double* k_LDA = new double[4];

  k_LDA[0] = 4 * M_PI;
  k_LDA[1] = 0.;
  k_LDA[2] = 0.;
  k_LDA[3] = 4 * M_PI / std::sqrt(3.);

  return k_LDA;
}

template <typename DCA_point_group_type>
std::vector<int> graphene_model<DCA_point_group_type>::get_flavors() {
  std::vector<int> flavors(BANDS);

  flavors[0] = 0;
  flavors[1] = 0;

  return flavors;
}

template <typename DCA_point_group_type>
std::vector<std::vector<double>> graphene_model<DCA_point_group_type>::get_a_vectors() {
  std::vector<std::vector<double>> a_vecs(0);

  {
    std::vector<double> a0(DIMENSION, 0.);

    a0[0] = 0.;
    a0[1] = 0.;

    a_vecs.push_back(a0);
  }

  {
    std::vector<double> a1(DIMENSION, 0.);

    a1[0] = 1.;
    a1[1] = 0.;

    a_vecs.push_back(a1);
  }

  return a_vecs;
}

template <typename DCA_point_group_type>
template <class domain, class parameters_type>
void graphene_model<DCA_point_group_type>::initialize_H_interaction(
    FUNC_LIB::function<double, domain>& H_interaction, parameters_type& parameters) {
  std::vector<std::vector<double>>& U_ij = parameters.get_U_ij();

  for (size_t i = 0; i < U_ij.size(); i++)
    H_interaction(U_ij[i][0], U_ij[i][1], U_ij[i][2], U_ij[i][3], U_ij[i][4]) = U_ij[i][5];
}

template <typename DCA_point_group_type>
std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> graphene_model<
    DCA_point_group_type>::get_orbital_permutations() {
  std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> permutations(0);

  {  // permutation 0
    std::pair<int, int> initial_state(0, 1);
    std::pair<int, int> final_state(1, 0);
    std::pair<std::pair<int, int>, std::pair<int, int>> p(initial_state, final_state);

    permutations.push_back(p);
  }

  return permutations;
}

template <typename DCA_point_group_type>
template <class domain>
void graphene_model<DCA_point_group_type>::initialize_H_symmetry(
    FUNC_LIB::function<int, domain>& H_symmetries) {
  // e_up <==> e_dn symmetry
  //   for(int i=0; i<BANDS; i++){
  //     for(int j=0; j<BANDS; j++){
  //       int l = j+i*BANDS;
  //       H_symmetries(i,0,j,0) = l;
  //       H_symmetries(i,1,j,1) = l;
  //     }
  //   }

  for (int i = 0; i < BANDS; i++) {
    H_symmetries(i, 0, i, 0) = 0;
    H_symmetries(i, 1, i, 1) = 0;
  }

  H_symmetries(0, 0, 1, 0) = 1;
  H_symmetries(0, 1, 1, 1) = 1;
  H_symmetries(1, 0, 0, 0) = 1;
  H_symmetries(1, 1, 0, 1) = 1;
}

template <typename DCA_point_group_type>
template <class parameters_type>
std::complex<double> graphene_model<DCA_point_group_type>::get_LDA_Hamiltonians(
    parameters_type& parameters, std::vector<double> k, int b1, int s1, int b2, int s2) {
  std::vector<std::vector<double>>& t_ij = parameters.get_t_ij();

  double t = 0;
  for (size_t i = 0; i < t_ij.size(); i++)
    if (t_ij[i][0] == b1 && t_ij[i][1] == b2 && t_ij[i][2] == 0)
      t = t_ij[i][3];

  std::complex<double> i(0., 1.);
  std::complex<double> H_LDA = 0;

  if (s1 == s2) {
    if (b1 == 0 && b2 == 0)
      H_LDA = t * (0.);

    if (b1 == 0 && b2 == 1)
      H_LDA = t * (0. + std::exp(i * (k[0] * (-1.0) + k[1] * (0.0))) +
                   std::exp(i * (k[0] * (0.5) + k[1] * (-0.866025403784))) +
                   std::exp(i * (k[0] * (0.5) + k[1] * (0.866025403784))));

    if (b1 == 1 && b2 == 0)
      H_LDA = t * (0. + std::exp(i * (k[0] * (-0.5) + k[1] * (-0.866025403784))) +
                   std::exp(i * (k[0] * (-0.5) + k[1] * (0.866025403784))) +
                   std::exp(i * (k[0] * (1.0) + k[1] * (0.0))));

    if (b1 == 1 && b2 == 1)
      H_LDA = t * (0.);
  }

  return H_LDA;
}

#endif  // PHYS_LIBRARY_PARAMETERS_MODELS_ANALYTIC_HAMILTONIANS_LATTICES_2D_HONEYCOMB_LATTICE_H
