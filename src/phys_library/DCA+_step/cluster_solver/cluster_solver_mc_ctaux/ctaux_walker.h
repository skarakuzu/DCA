//-*-C++-*-

#ifndef DCA_QMCI_CT_AUX_WALKER_H
#define DCA_QMCI_CT_AUX_WALKER_H

#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_template/qmci_walker.h"
#include "dca/math_library/random_number_library/random_number_library.hpp"
#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_ctaux/ctaux_structs/ctaux_walker_data.h"
#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_ctaux/ctaux_structs/ctaux_vertex_singleton.h"
#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_ctaux/ctaux_structs/ctaux_hubbard_stratonovitch_configuration.h"
#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_ctaux/ctaux_typedefinitions.h"
#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_ctaux/ctaux_domains/HS_vertex_move_domain.h"
#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_ctaux/ctaux_walker/ctaux_walker_routines_CPU.h"
#ifdef USE_GPU
#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_ctaux/ctaux_walker/ctaux_walker_routines_GPU.h"
#endif
#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_ctaux/ctaux_walker/ctaux_walker_tools/ctaux_walker_tools.hpp"
// TODO maybe. Include everything only if QMC_INTEGRATOR_BIT is defined
#include "phys_library/DCA+_step/cluster_solver/cluster_solver_mc_ctaux/ctaux_walker/ctaux_walker_build_in_test.h"

namespace DCA {
namespace QMCI {
/*!
 *  \defgroup CT-AUX-WALKER
 *  \ingroup  CT-AUX
 */

/*!
 *  \defgroup STRUCTURES
 *  \ingroup  CT-AUX
 */

/*!
 *  \ingroup CT-AUX
 *
 *  \brief   This class organizes the MC-walker in the CT-AUX QMC
 *  \author  Peter Staar
 *  \version 1.0
 */
template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
class MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>
    : public MC_walker_BIT<CT_AUX_SOLVER, parameters_type, MOMS_type>,
      public MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type> {
  typedef vertex_singleton vertex_singleton_type;
  typedef CT_AUX_HS_configuration<parameters_type> configuration_type;

  using rng_type = typename parameters_type::random_number_generator;

  typedef
      typename MC_type_definitions<CT_AUX_SOLVER, parameters_type, MOMS_type>::profiler_type profiler_type;
  typedef typename MC_type_definitions<CT_AUX_SOLVER, parameters_type, MOMS_type>::concurrency_type
      concurrency_type;

public:
  const static LIN_ALG::device_type walker_device_type = device_t;

public:
  MC_walker(parameters_type& parameters_ref, MOMS_type& MOMS_ref, rng_type& rng_ref, int id);

  ~MC_walker();

  void initialize();

  bool& is_thermalized();

  void do_sweep();
  void do_step();

  LIN_ALG::matrix<double, device_t>& get_N(e_spin_states_type e_spin);
  configuration_type& get_configuration();

  int get_sign();
  int get_thread_id();

  double get_Gflop();

  template <class stream_type>
  void to_JSON(stream_type& ss);

private:
  void add_non_interacting_spins_to_configuration();

  void generate_delayed_spins();
  void read_Gamma_matrices(e_spin_states e_spin);
  void compute_Gamma_matrices();

  void add_delayed_spin(int& delayed_index, int& Gamma_up_size, int& Gamma_dn_size);

  void add_delayed_spins_to_the_configuration();
  void remove_non_accepted_and_bennett_spins_from_Gamma(int& Gamma_up_size, int& Gamma_dn_size);

  void apply_bennett_on_Gamma_matrices(int& Gamma_up_size, int& Gamma_dn_size);
  void neutralize_delayed_spin(int& delayed_index, int& Gamma_up_size, int& Gamma_dn_size);

  void download_from_device();
  void upload_to_device();

  void add_delayed_spins();
  void update_Gamma_matrix_spin_by_spin();

  void update_N_matrix_with_Gamma_matrix();

  void clean_up_the_configuration();

  bool store_Gamma(HS_vertex_move_type HS_current_move, int random_vertex_ind);

  void restore_Gamma(int random_vertex_ind);

  HS_vertex_move_type get_new_HS_move();
  int get_new_vertex_index(HS_vertex_move_type HS_current_move);
  HS_spin_states_type get_new_spin_value(HS_vertex_move_type HS_current_move);

  double compute_determinant_ratio(int random_vertex_ind, HS_spin_states_type new_HS_spin_value,
                                   bool Bennett);
  double calculate_acceptace_ratio(double ratio, HS_vertex_move_type HS_current_move,
                                   double QMC_factor);

  bool assert_exp_delta_V_value(HS_field_sign HS_field, int random_vertex_ind,
                                HS_spin_states_type new_HS_spin_value, double exp_delta_V);

private:
#ifdef QMC_INTEGRATOR_BIT
  using MC_walker_BIT<CT_AUX_SOLVER, parameters_type, MOMS_type>::check_G0_matrices;
  using MC_walker_BIT<CT_AUX_SOLVER, parameters_type, MOMS_type>::check_N_matrices;
  using MC_walker_BIT<CT_AUX_SOLVER, parameters_type, MOMS_type>::check_G_matrices;
#endif

private:
  struct delayed_spin_struct {
    int delayed_spin_index;

    HS_vertex_move_type HS_current_move;
    int random_vertex_ind;
    HS_spin_states_type new_HS_spin_value;

    double QMC_factor;

    bool is_accepted_move;
    bool is_a_bennett_spin;

    e_spin_states e_spin_HS_field_DN;
    e_spin_states e_spin_HS_field_UP;

    int configuration_e_spin_index_HS_field_DN;
    int configuration_e_spin_index_HS_field_UP;

    int Gamma_index_HS_field_DN;
    int Gamma_index_HS_field_UP;

    double exp_V_HS_field_DN;
    double exp_V_HS_field_UP;

    double exp_delta_V_HS_field_DN;
    double exp_delta_V_HS_field_UP;

    double exp_minus_delta_V_HS_field_UP;
    double exp_minus_delta_V_HS_field_DN;
  };

private:
  parameters_type& parameters;
  MOMS_type& MOMS;
  concurrency_type& concurrency;

  int thread_id;
  int stream_id;

  CV<parameters_type> CV_obj;
  CT_AUX_WALKER_TOOLS<LIN_ALG::CPU> ctaux_tools;

  rng_type& rng;
  configuration_type configuration;

  G0_INTERPOLATION<device_t, parameters_type> G0_tools_obj;
  N_TOOLS<device_t, parameters_type> N_tools_obj;
  G_TOOLS<device_t, parameters_type> G_tools_obj;

  SHRINK_TOOLS<device_t> SHRINK_tools_obj;

  using MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>::N_up;
  using MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>::N_dn;

  using MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>::G0_up;
  using MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>::G0_dn;

  using MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>::Gamma_up;
  using MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>::Gamma_dn;

  using MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>::G_up;
  using MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>::G_dn;

  LIN_ALG::matrix<double, LIN_ALG::CPU> Gamma_up_CPU;
  LIN_ALG::matrix<double, LIN_ALG::CPU> Gamma_dn_CPU;

  double Gamma_up_diag_max;
  double Gamma_up_diag_min;
  double Gamma_dn_diag_max;
  double Gamma_dn_diag_min;

  LIN_ALG::matrix<double, LIN_ALG::CPU> stored_Gamma_up_CPU;
  LIN_ALG::matrix<double, LIN_ALG::CPU> stored_Gamma_dn_CPU;

  std::vector<int> random_vertex_vector;
  std::vector<HS_vertex_move_type> HS_current_move_vector;
  std::vector<HS_spin_states_type> new_HS_spin_value_vector;

  std::vector<int> vertex_indixes_CPU;
  std::vector<double> exp_V_CPU;
  std::vector<double> exp_delta_V_CPU;

  LIN_ALG::vector<int, device_t> vertex_indixes;
  LIN_ALG::vector<double, device_t> exp_V;
  LIN_ALG::vector<double, device_t> exp_delta_V;

  std::vector<delayed_spin_struct> delayed_spins;
  std::vector<delayed_spin_struct> bennett_spins;

  int number_of_interacting_spins;

  int number_of_creations;
  int number_of_annihilations;

  bool thermalized;
  bool Bennett;

  int sign;
};

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::MC_walker(
    parameters_type& parameters_ref, MOMS_type& MOMS_ref, rng_type& rng_ref, int id)
    : MC_walker_BIT<CT_AUX_SOLVER, parameters_type, MOMS_type>(parameters_ref, MOMS_ref, id),

      MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>(parameters_ref, id),

      parameters(parameters_ref),
      MOMS(MOMS_ref),
      concurrency(parameters.get_concurrency()),

      thread_id(id),
      stream_id(0),

      CV_obj(parameters),
      ctaux_tools(MC_walker_data<CT_AUX_SOLVER, device_t, parameters_type>::MAX_VERTEX_SINGLETS *
                  parameters.get_K_PHANI()),

      rng(rng_ref),

      configuration(parameters, rng),

      G0_tools_obj(thread_id, parameters),
      N_tools_obj(thread_id, parameters, CV_obj),
      G_tools_obj(thread_id, parameters, CV_obj),

      SHRINK_tools_obj(thread_id),

      Gamma_up_CPU("Gamma_up_CPU", Gamma_up.get_current_size(), Gamma_up.get_global_size()),
      Gamma_dn_CPU("Gamma_dn_CPU", Gamma_dn.get_current_size(), Gamma_dn.get_global_size()),

      Gamma_up_diag_max(1),
      Gamma_up_diag_min(1),
      Gamma_dn_diag_max(1),
      Gamma_dn_diag_min(1),

      stored_Gamma_up_CPU("stored_Gamma_up_CPU", Gamma_up.get_current_size(),
                          Gamma_up.get_global_size()),
      stored_Gamma_dn_CPU("stored_Gamma_dn_CPU", Gamma_dn.get_current_size(),
                          Gamma_dn.get_global_size()),

      random_vertex_vector(0),
      HS_current_move_vector(0),
      new_HS_spin_value_vector(0),

      thermalized(false),
      Bennett(false),
      sign(1) {
  if (concurrency.id() == 0 and thread_id == 0) {
    std::cout << "\n\n"
              << "\t\t"
              << "CT-AUX walker test"
              << "\n\n";
  }

  LIN_ALG::CUBLAS_THREAD_MANAGER<device_t>::initialize(thread_id);

  Gamma_up_CPU.set_thread_and_stream_id(thread_id, 0);
  Gamma_dn_CPU.set_thread_and_stream_id(thread_id, 0);

  stored_Gamma_up_CPU.set_thread_and_stream_id(thread_id, 0);
  stored_Gamma_dn_CPU.set_thread_and_stream_id(thread_id, 0);

  vertex_indixes.set_thread_and_stream_id(thread_id, 0);
  exp_V.set_thread_and_stream_id(thread_id, 0);
  exp_delta_V.set_thread_and_stream_id(thread_id, 0);
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::~MC_walker() {
  LIN_ALG::CUBLAS_THREAD_MANAGER<device_t>::finalize(thread_id);

  if (concurrency.id() == 0 and thread_id == 0) {
    std::cout << "\n\n\t"  //<< number_of_creations << "\t" << number_of_annihilations
              << "\t\t creations/annihilations : "
              << double(number_of_creations) / double(number_of_annihilations) << "\n\n";
  }
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
LIN_ALG::matrix<double, device_t>& MC_walker<CT_AUX_SOLVER, device_t, parameters_type,
                                             MOMS_type>::get_N(e_spin_states_type e_spin) {
  if (e_spin == e_DN)
    return N_dn;
  else
    return N_up;
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
typename MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::configuration_type& MC_walker<
    CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::get_configuration() {
  return configuration;
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
int MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::get_sign() {
  return sign;
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
int MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::get_thread_id() {
  return thread_id;
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
double MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::get_Gflop() {
  double Gflop = 0.;

  Gflop += N_tools_obj.get_Gflop();
  Gflop += G_tools_obj.get_Gflop();

  return Gflop;
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
bool& MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::is_thermalized() {
  return thermalized;
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
template <class stream_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::to_JSON(stream_type& /*ss*/) {}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::initialize() {
  MC_walker_BIT<CT_AUX_SOLVER, parameters_type, MOMS_type>::initialize();

  number_of_creations = 0;
  number_of_annihilations = 0;

  sign = 1;

  CV_obj.initialize(MOMS);

  configuration.initialize();

  is_thermalized() = false;

  {
    G0_tools_obj.initialize(MOMS);
    G0_tools_obj.build_G0_matrix(configuration, G0_up, e_UP);
    G0_tools_obj.build_G0_matrix(configuration, G0_dn, e_DN);
#ifdef QMC_INTEGRATOR_BIT
    check_G0_matrices(configuration, G0_up, G0_dn);
#endif
  }

  {
    N_tools_obj.build_N_matrix(configuration, N_up, G0_up, e_UP);
    N_tools_obj.build_N_matrix(configuration, N_dn, G0_dn, e_DN);

#ifdef QMC_INTEGRATOR_BIT
    check_N_matrices(configuration, G0_up, G0_dn, N_up, N_dn);
#endif
  }

  {
    G_tools_obj.build_G_matrix(configuration, N_up, G0_up, G_up, e_UP);
    G_tools_obj.build_G_matrix(configuration, N_dn, G0_dn, G_dn, e_DN);

#ifdef QMC_INTEGRATOR_BIT
    check_G_matrices(configuration, G0_up, G0_dn, N_up, N_dn, G_up, G_dn);
#endif
  }
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::do_sweep() {
  double factor = 1.;
  if (thermalized)
    factor = parameters.get_number_of_sweeps_per_measurement();

  int nb_of_block_steps = 1 + floor(configuration.get_number_of_interacting_HS_spins() /
                                    parameters.get_K_PHANI() * factor);

  for (int i = 0; i < nb_of_block_steps; i++)
    do_step();
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::do_step() {
  add_non_interacting_spins_to_configuration();

  {
    generate_delayed_spins();

    download_from_device();

    compute_Gamma_matrices();

    upload_to_device();
  }

  update_N_matrix_with_Gamma_matrix();

  clean_up_the_configuration();
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::download_from_device() {
  profiler_type profiler(__FUNCTION__, "CT-AUX walker", __LINE__, thread_id);

  assert(Gamma_up_CPU.get_global_size() == Gamma_up.get_global_size());
  assert(Gamma_dn_CPU.get_global_size() == Gamma_dn.get_global_size());

  read_Gamma_matrices(e_UP);
  read_Gamma_matrices(e_DN);

  if (device_t == LIN_ALG::CPU) {
    Gamma_up_CPU.swap(Gamma_up);
    Gamma_dn_CPU.swap(Gamma_dn);
  }
  else {
    Gamma_up_CPU.copy_from(Gamma_up, LIN_ALG::ASYNCHRONOUS);
    Gamma_dn_CPU.copy_from(Gamma_dn, LIN_ALG::ASYNCHRONOUS);
  }
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::upload_to_device() {
  profiler_type profiler(__FUNCTION__, "CT-AUX walker", __LINE__, thread_id);

  assert(Gamma_up_CPU.get_global_size() == Gamma_up.get_global_size());
  assert(Gamma_dn_CPU.get_global_size() == Gamma_dn.get_global_size());

  if (device_t == LIN_ALG::CPU) {
    Gamma_up.swap(Gamma_up_CPU);
    Gamma_dn.swap(Gamma_dn_CPU);
  }
  else {
    Gamma_up.copy_from(Gamma_up_CPU, LIN_ALG::ASYNCHRONOUS);
    Gamma_dn.copy_from(Gamma_dn_CPU, LIN_ALG::ASYNCHRONOUS);
  }
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type,
               MOMS_type>::add_non_interacting_spins_to_configuration() {
  // profiler_type profiler(__FUNCTION__, "CT-AUX walker", __LINE__, thread_id);

  Gamma_up.resize_no_copy(0);
  Gamma_dn.resize_no_copy(0);

  // shuffle the configuration + do some configuration checks
  configuration.shuffle_noninteracting_vertices();

  {  // update G0 for new shuffled vertices
    profiler_type profiler("G0-matrix (update)", "CT-AUX walker", __LINE__, thread_id);

    G0_tools_obj.update_G0_matrix(configuration, G0_up, e_UP);
    G0_tools_obj.update_G0_matrix(configuration, G0_dn, e_DN);

#ifdef QMC_INTEGRATOR_BIT
    check_G0_matrices(configuration, G0_up, G0_dn);
#endif
  }

  {  // update N for new shuffled vertices
    profiler_type profiler("N-matrix (update)", "CT-AUX walker", __LINE__, thread_id);

    N_tools_obj.update_N_matrix(configuration, G0_up, N_up, e_UP);
    N_tools_obj.update_N_matrix(configuration, G0_dn, N_dn, e_DN);

#ifdef QMC_INTEGRATOR_BIT
    check_N_matrices(configuration, G0_up, G0_dn, N_up, N_dn);
#endif
  }

  {  // update N for new shuffled vertices
    profiler_type profiler("G-matrix (update)", "CT-AUX walker", __LINE__, thread_id);

    G_tools_obj.build_G_matrix(configuration, N_up, G0_up, G_up, e_UP);
    G_tools_obj.build_G_matrix(configuration, N_dn, G0_dn, G_dn, e_DN);

#ifdef QMC_INTEGRATOR_BIT
    check_G_matrices(configuration, G0_up, G0_dn, N_up, N_dn, G_up, G_dn);
#endif
  }
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::generate_delayed_spins() {
  profiler_type profiler(__FUNCTION__, "CT-AUX walker", __LINE__, thread_id);

  delayed_spins.resize(0);

  int N_c = 0;
  int N_a = 0;

  int N_s = configuration.size();
  int N_i = configuration.get_number_of_interacting_HS_spins();

  int number_of_delayed_spins = 0;
  if (is_thermalized()) {
    int new_size = configuration.size() / 4;
    number_of_delayed_spins = std::min((new_size / 32 + 1) * 32, 2 * parameters.get_K_PHANI());
  }
  else
    number_of_delayed_spins = 2 * parameters.get_K_PHANI();

  for (int i = 0; i < number_of_delayed_spins; i++) {
    delayed_spin_struct delayed_spin;

    delayed_spin.is_accepted_move = false;
    delayed_spin.is_a_bennett_spin = false;

    delayed_spin.HS_current_move = get_new_HS_move();

    if (delayed_spin.HS_current_move != STATIC) {
      bool has_already_been_chosen = true;

      while (has_already_been_chosen) {
        delayed_spin.random_vertex_ind = get_new_vertex_index(delayed_spin.HS_current_move);
        delayed_spin.new_HS_spin_value = get_new_spin_value(delayed_spin.HS_current_move);

        has_already_been_chosen = false;
        for (size_t l = 0; l < delayed_spins.size(); ++l)
          if (delayed_spin.random_vertex_ind == delayed_spins[l].random_vertex_ind)
            has_already_been_chosen = true;
      }

      assert(!has_already_been_chosen);

      delayed_spins.push_back(delayed_spin);

      if (delayed_spin.HS_current_move == CREATION)
        N_c += 1;

      if (delayed_spin.HS_current_move == ANNIHILATION)
        N_a += 1;

      if ((N_c == N_s - N_i) or (N_a == N_i and N_i > 0))
        break;
    }
  }

  int Gamma_dn_size = 0;
  int Gamma_up_size = 0;

  for (size_t i = 0; i < delayed_spins.size(); ++i) {
    // std::cout << "\t" << i << "\t";

    delayed_spins[i].delayed_spin_index = i;

    int random_vertex_ind = delayed_spins[i].random_vertex_ind;
    HS_spin_states_type new_HS_spin_value = delayed_spins[i].new_HS_spin_value;

    delayed_spins[i].QMC_factor =
        CV_obj.get_QMC_factor(configuration[random_vertex_ind], new_HS_spin_value);

    delayed_spins[i].e_spin_HS_field_DN = configuration[random_vertex_ind].get_e_spins().first;
    delayed_spins[i].e_spin_HS_field_UP = configuration[random_vertex_ind].get_e_spins().second;

    delayed_spins[i].configuration_e_spin_index_HS_field_DN =
        configuration[random_vertex_ind].get_configuration_e_spin_indices().first;
    delayed_spins[i].configuration_e_spin_index_HS_field_UP =
        configuration[random_vertex_ind].get_configuration_e_spin_indices().second;

    if (delayed_spins[i].e_spin_HS_field_DN == e_UP) {
      delayed_spins[i].Gamma_index_HS_field_DN = Gamma_up_size;
      Gamma_up_size += 1;

      vertex_singleton_type& v_j =
          configuration.get(e_UP)[delayed_spins[i].configuration_e_spin_index_HS_field_DN];

      delayed_spins[i].exp_V_HS_field_DN = CV_obj.exp_V(v_j);
      delayed_spins[i].exp_delta_V_HS_field_DN = CV_obj.exp_delta_V(v_j, new_HS_spin_value);
      delayed_spins[i].exp_minus_delta_V_HS_field_DN =
          CV_obj.exp_minus_delta_V(v_j, new_HS_spin_value);
    }
    else {
      delayed_spins[i].Gamma_index_HS_field_DN = Gamma_dn_size;
      Gamma_dn_size += 1;

      vertex_singleton_type& v_j =
          configuration.get(e_DN)[delayed_spins[i].configuration_e_spin_index_HS_field_DN];

      delayed_spins[i].exp_V_HS_field_DN = CV_obj.exp_V(v_j);
      delayed_spins[i].exp_delta_V_HS_field_DN = CV_obj.exp_delta_V(v_j, new_HS_spin_value);
      delayed_spins[i].exp_minus_delta_V_HS_field_DN =
          CV_obj.exp_minus_delta_V(v_j, new_HS_spin_value);
    }

    if (delayed_spins[i].e_spin_HS_field_UP == e_UP) {
      delayed_spins[i].Gamma_index_HS_field_UP = Gamma_up_size;
      Gamma_up_size += 1;

      vertex_singleton_type& v_j =
          configuration.get(e_UP)[delayed_spins[i].configuration_e_spin_index_HS_field_UP];

      delayed_spins[i].exp_V_HS_field_UP = CV_obj.exp_V(v_j);
      delayed_spins[i].exp_delta_V_HS_field_UP = CV_obj.exp_delta_V(v_j, new_HS_spin_value);
      delayed_spins[i].exp_minus_delta_V_HS_field_UP =
          CV_obj.exp_minus_delta_V(v_j, new_HS_spin_value);
    }
    else {
      delayed_spins[i].Gamma_index_HS_field_UP = Gamma_dn_size;
      Gamma_dn_size += 1;

      vertex_singleton_type& v_j =
          configuration.get(e_DN)[delayed_spins[i].configuration_e_spin_index_HS_field_UP];

      delayed_spins[i].exp_V_HS_field_UP = CV_obj.exp_V(v_j);
      delayed_spins[i].exp_delta_V_HS_field_UP = CV_obj.exp_delta_V(v_j, new_HS_spin_value);
      delayed_spins[i].exp_minus_delta_V_HS_field_UP =
          CV_obj.exp_minus_delta_V(v_j, new_HS_spin_value);
    }
  }
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::read_Gamma_matrices(
    e_spin_states e_spin) {
  // profiler_type profiler(concurrency, __FUNCTION__, "CT-AUX walker", __LINE__, thread_id);

  {
    exp_V_CPU.resize(0);
    exp_delta_V_CPU.resize(0);
    vertex_indixes_CPU.resize(0);

    for (size_t i = 0; i < delayed_spins.size(); ++i) {
      if (delayed_spins[i].e_spin_HS_field_DN == e_spin) {
        exp_V_CPU.push_back(delayed_spins[i].exp_V_HS_field_DN);
        exp_delta_V_CPU.push_back(delayed_spins[i].exp_delta_V_HS_field_DN);
        vertex_indixes_CPU.push_back(delayed_spins[i].configuration_e_spin_index_HS_field_DN);
      }

      if (delayed_spins[i].e_spin_HS_field_UP == e_spin) {
        exp_V_CPU.push_back(delayed_spins[i].exp_V_HS_field_UP);
        exp_delta_V_CPU.push_back(delayed_spins[i].exp_delta_V_HS_field_UP);
        vertex_indixes_CPU.push_back(delayed_spins[i].configuration_e_spin_index_HS_field_UP);
      }
    }

    vertex_indixes.set(vertex_indixes_CPU, LIN_ALG::ASYNCHRONOUS);
    exp_V.set(exp_V_CPU, LIN_ALG::ASYNCHRONOUS);
    exp_delta_V.set(exp_delta_V_CPU, LIN_ALG::ASYNCHRONOUS);
  }

  switch (e_spin) {
    case e_DN:
      CT_AUX_WALKER_TOOLS<device_t>::compute_Gamma(Gamma_dn, N_dn, G_dn, vertex_indixes, exp_V,
                                                   exp_delta_V, thread_id, stream_id);
      break;

    case e_UP:
      CT_AUX_WALKER_TOOLS<device_t>::compute_Gamma(Gamma_up, N_up, G_up, vertex_indixes, exp_V,
                                                   exp_delta_V, thread_id, stream_id);
      break;

    default:
      throw std::logic_error(__FUNCTION__);
  }
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::compute_Gamma_matrices() {
  profiler_type profiler(__FUNCTION__, "CT-AUX walker", __LINE__, thread_id);

  bennett_spins.resize(0);

  int Gamma_up_size = 0;
  int Gamma_dn_size = 0;

  Gamma_up_diag_max = 1;
  Gamma_up_diag_min = 1;
  Gamma_dn_diag_max = 1;
  Gamma_dn_diag_min = 1;

  number_of_interacting_spins = configuration.get_number_of_interacting_HS_spins();

  for (int delayed_index = 0; delayed_index < int(delayed_spins.size()); delayed_index++) {
    if (delayed_spins[delayed_index].HS_current_move == ANNIHILATION) {
      double alpha = 0;
      if (number_of_interacting_spins > 0)
        alpha = double(bennett_spins.size()) / double(number_of_interacting_spins);

      if (false and rng.get_random_number() < alpha) {
        apply_bennett_on_Gamma_matrices(Gamma_up_size, Gamma_dn_size);

        neutralize_delayed_spin(delayed_index, Gamma_up_size, Gamma_dn_size);
      }
      else {
        add_delayed_spin(delayed_index, Gamma_up_size, Gamma_dn_size);
      }
    }
    else {
      add_delayed_spin(delayed_index, Gamma_up_size, Gamma_dn_size);
    }
  }

  add_delayed_spins_to_the_configuration();

  remove_non_accepted_and_bennett_spins_from_Gamma(Gamma_up_size, Gamma_dn_size);
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::neutralize_delayed_spin(
    int& delayed_index, int& Gamma_up_size, int& Gamma_dn_size) {
  delayed_spins[delayed_index].is_accepted_move = false;

  if (delayed_spins[delayed_index].e_spin_HS_field_DN == e_UP) {
    Gamma_up_size += 1;
    CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(
        Gamma_up_CPU, delayed_spins[delayed_index].Gamma_index_HS_field_DN);
  }
  else {
    Gamma_dn_size += 1;
    CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(
        Gamma_dn_CPU, delayed_spins[delayed_index].Gamma_index_HS_field_DN);
  }

  if (delayed_spins[delayed_index].e_spin_HS_field_UP == e_UP) {
    Gamma_up_size += 1;
    CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(
        Gamma_up_CPU, delayed_spins[delayed_index].Gamma_index_HS_field_UP);
  }
  else {
    Gamma_dn_size += 1;
    CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(
        Gamma_dn_CPU, delayed_spins[delayed_index].Gamma_index_HS_field_UP);
  }
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type,
               MOMS_type>::add_delayed_spins_to_the_configuration() {
  for (size_t i = 0; i < delayed_spins.size(); ++i) {
    int configuration_index = delayed_spins[i].random_vertex_ind;

    if (delayed_spins[i].is_accepted_move) {
      if (not delayed_spins[i].is_a_bennett_spin) {
        configuration.add_delayed_HS_spin(configuration_index, delayed_spins[i].new_HS_spin_value);
      }
      else {
        configuration[configuration_index].is_creatable() = false;
        configuration[configuration_index].is_annihilatable() = false;
      }
    }
  }

  assert(number_of_interacting_spins == configuration.get_number_of_interacting_HS_spins());
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type,
               MOMS_type>::remove_non_accepted_and_bennett_spins_from_Gamma(int& Gamma_up_size,
                                                                            int& Gamma_dn_size) {
  // std::cout << __FUNCTION__ << "\n";

  for (int i = delayed_spins.size() - 1; i > -1; i--) {
    if ((not delayed_spins[i].is_accepted_move) or delayed_spins[i].is_a_bennett_spin) {
      e_spin_states e_spin_HS_field_DN = delayed_spins[i].e_spin_HS_field_DN;
      e_spin_states e_spin_HS_field_UP = delayed_spins[i].e_spin_HS_field_UP;

      int Gamma_index_HS_field_UP = delayed_spins[i].Gamma_index_HS_field_UP;
      int Gamma_index_HS_field_DN = delayed_spins[i].Gamma_index_HS_field_DN;

      if (e_spin_HS_field_UP == e_UP) {
        Gamma_up_size -= 1;
        CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::remove_row_and_column(Gamma_up_CPU,
                                                                 Gamma_index_HS_field_UP);
      }
      else {
        Gamma_dn_size -= 1;
        CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::remove_row_and_column(Gamma_dn_CPU,
                                                                 Gamma_index_HS_field_UP);
      }

      if (e_spin_HS_field_DN == e_UP) {
        Gamma_up_size -= 1;
        CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::remove_row_and_column(Gamma_up_CPU,
                                                                 Gamma_index_HS_field_DN);
      }
      else {
        Gamma_dn_size -= 1;
        CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::remove_row_and_column(Gamma_dn_CPU,
                                                                 Gamma_index_HS_field_DN);
      }
    }
  }

  assert(Gamma_up_size == Gamma_up_CPU.get_current_size().first and
         Gamma_up_size == Gamma_up_CPU.get_current_size().second);
  assert(Gamma_dn_size == Gamma_dn_CPU.get_current_size().first and
         Gamma_dn_size == Gamma_dn_CPU.get_current_size().second);
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::add_delayed_spin(
    int& delayed_index, int& Gamma_up_size, int& Gamma_dn_size) {
  assert(assert_exp_delta_V_value(HS_FIELD_DN, delayed_spins[delayed_index].random_vertex_ind,
                                  delayed_spins[delayed_index].new_HS_spin_value,
                                  delayed_spins[delayed_index].exp_delta_V_HS_field_DN));

  assert(assert_exp_delta_V_value(HS_FIELD_UP, delayed_spins[delayed_index].random_vertex_ind,
                                  delayed_spins[delayed_index].new_HS_spin_value,
                                  delayed_spins[delayed_index].exp_delta_V_HS_field_UP));

  if (delayed_spins[delayed_index].HS_current_move == CREATION)
    number_of_creations += 1;

  if (delayed_spins[delayed_index].HS_current_move == ANNIHILATION)
    number_of_annihilations += 1;

  int Gamma_index_HS_field_DN = delayed_spins[delayed_index].Gamma_index_HS_field_DN;  //-1;
  int Gamma_index_HS_field_UP = delayed_spins[delayed_index].Gamma_index_HS_field_UP;  //-1;

  double exp_delta_V_HS_field_DN = delayed_spins[delayed_index].exp_delta_V_HS_field_DN;
  double exp_delta_V_HS_field_UP = delayed_spins[delayed_index].exp_delta_V_HS_field_UP;

  double ratio_HS_field_DN = 0;
  double ratio_HS_field_UP = 0;

  double tmp_up_diag_max = Gamma_up_diag_max;
  double tmp_up_diag_min = Gamma_up_diag_min;
  double tmp_dn_diag_max = Gamma_dn_diag_max;
  double tmp_dn_diag_min = Gamma_dn_diag_min;

  {
    if (delayed_spins[delayed_index].e_spin_HS_field_DN == e_UP) {
      assert(Gamma_index_HS_field_DN == Gamma_up_size);

      ratio_HS_field_DN = ctaux_tools.solve_Gamma_blocked(Gamma_index_HS_field_DN, Gamma_up_CPU,
                                                          exp_delta_V_HS_field_DN,
                                                          Gamma_up_diag_max, Gamma_up_diag_min);

      Gamma_up_size += 1;
    }
    else {
      assert(Gamma_index_HS_field_DN == Gamma_dn_size);

      ratio_HS_field_DN = ctaux_tools.solve_Gamma_blocked(Gamma_index_HS_field_DN, Gamma_dn_CPU,
                                                          exp_delta_V_HS_field_DN,
                                                          Gamma_dn_diag_max, Gamma_dn_diag_min);

      Gamma_dn_size += 1;
    }

    if (delayed_spins[delayed_index].e_spin_HS_field_UP == e_UP) {
      assert(Gamma_index_HS_field_UP == Gamma_up_size);

      ratio_HS_field_UP = ctaux_tools.solve_Gamma_blocked(Gamma_index_HS_field_UP, Gamma_up_CPU,
                                                          exp_delta_V_HS_field_UP,
                                                          Gamma_up_diag_max, Gamma_up_diag_min);

      Gamma_up_size += 1;
    }
    else {
      assert(Gamma_index_HS_field_UP == Gamma_dn_size);

      ratio_HS_field_UP = ctaux_tools.solve_Gamma_blocked(Gamma_index_HS_field_UP, Gamma_dn_CPU,
                                                          exp_delta_V_HS_field_UP,
                                                          Gamma_dn_diag_max, Gamma_dn_diag_min);

      Gamma_dn_size += 1;
    }
  }

  double determinant_ratio = ratio_HS_field_UP * ratio_HS_field_DN;
  double acceptance_ratio =
      calculate_acceptace_ratio(determinant_ratio, delayed_spins[delayed_index].HS_current_move,
                                delayed_spins[delayed_index].QMC_factor);

  if (std::fabs(acceptance_ratio) >= rng.get_random_number()) {
    delayed_spins[delayed_index].is_accepted_move = true;

    if (acceptance_ratio < 0)
      sign *= -1;

    assert(delayed_spins[delayed_index].delayed_spin_index == delayed_index);

    if (delayed_spins[delayed_index].HS_current_move == CREATION) {
      bennett_spins.push_back(delayed_spins[delayed_index]);

      bennett_spins.back().HS_current_move = ANNIHILATION;
    }

    if (delayed_spins[delayed_index].HS_current_move == CREATION)
      number_of_interacting_spins += 1;

    if (delayed_spins[delayed_index].HS_current_move == ANNIHILATION)
      number_of_interacting_spins -= 1;
  }
  else {
    if (delayed_spins[delayed_index].e_spin_HS_field_DN == e_UP and
        delayed_spins[delayed_index].e_spin_HS_field_UP == e_UP) {
      Gamma_up_diag_max = tmp_up_diag_max < 1. ? 1. : tmp_up_diag_max;
      Gamma_up_diag_min = tmp_up_diag_min > 1. ? 1. : tmp_up_diag_min;

      CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_up_CPU, Gamma_up_size - 2);
      CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_up_CPU, Gamma_up_size - 1);
    }

    if (delayed_spins[delayed_index].e_spin_HS_field_DN == e_DN and
        delayed_spins[delayed_index].e_spin_HS_field_UP == e_UP) {
      Gamma_up_diag_max = tmp_up_diag_max < 1. ? 1. : tmp_up_diag_max;
      Gamma_dn_diag_max = tmp_dn_diag_max < 1. ? 1. : tmp_dn_diag_max;
      Gamma_up_diag_min = tmp_up_diag_min > 1. ? 1. : tmp_up_diag_min;
      Gamma_dn_diag_min = tmp_dn_diag_min > 1. ? 1. : tmp_dn_diag_min;

      CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_dn_CPU, Gamma_dn_size - 1);
      CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_up_CPU, Gamma_up_size - 1);
    }

    if (delayed_spins[delayed_index].e_spin_HS_field_DN == e_UP and
        delayed_spins[delayed_index].e_spin_HS_field_UP == e_DN) {
      Gamma_up_diag_max = tmp_up_diag_max < 1. ? 1. : tmp_up_diag_max;
      Gamma_dn_diag_max = tmp_dn_diag_max < 1. ? 1. : tmp_dn_diag_max;
      Gamma_up_diag_min = tmp_up_diag_min > 1. ? 1. : tmp_up_diag_min;
      Gamma_dn_diag_min = tmp_dn_diag_min > 1. ? 1. : tmp_dn_diag_min;

      CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_up_CPU, Gamma_up_size - 1);
      CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_dn_CPU, Gamma_dn_size - 1);
    }

    if (delayed_spins[delayed_index].e_spin_HS_field_DN == e_DN and
        delayed_spins[delayed_index].e_spin_HS_field_UP == e_DN) {
      Gamma_dn_diag_max = tmp_dn_diag_max < 1. ? 1. : tmp_dn_diag_max;
      Gamma_dn_diag_min = tmp_dn_diag_min > 1. ? 1. : tmp_dn_diag_min;

      CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_dn_CPU, Gamma_dn_size - 2);
      CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_dn_CPU, Gamma_dn_size - 1);
    }
  }
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::apply_bennett_on_Gamma_matrices(
    int& /*Gamma_up_size*/, int& /*Gamma_dn_size*/) {
  throw std::logic_error(__FUNCTION__);
  // TODO implement Bennet updates
  /*
    number_of_annihilations += 1;

    stored_Gamma_up_CPU.copy_from(Gamma_up_CPU);
    stored_Gamma_dn_CPU.copy_from(Gamma_dn_CPU);

    double ratio_HS_field_DN = 0;
    double ratio_HS_field_UP = 0;

    int bennett_index = int(rng.get_random_number()*double(bennett_spins.size()));

    e_spin_states e_spin_HS_field_DN = bennett_spins[bennett_index].e_spin_HS_field_DN;
    e_spin_states e_spin_HS_field_UP = bennett_spins[bennett_index].e_spin_HS_field_UP;

    if(e_spin_HS_field_DN == e_UP)
    {
    int    index             = bennett_spins[bennett_index].Gamma_index_HS_field_DN;
    double exp_minus_delta_V = bennett_spins[bennett_index].exp_minus_delta_V_HS_field_DN;

    ratio_HS_field_DN = ctaux_tools.apply_bennett_on_Gamma(index, Gamma_up_size, Gamma_up_CPU,
    exp_minus_delta_V);
    }
    else
    {
    int    index             = bennett_spins[bennett_index].Gamma_index_HS_field_DN;
    double exp_minus_delta_V = bennett_spins[bennett_index].exp_minus_delta_V_HS_field_DN;

    ratio_HS_field_DN = ctaux_tools.apply_bennett_on_Gamma(index, Gamma_dn_size, Gamma_dn_CPU,
    exp_minus_delta_V);
    }

    if(e_spin_HS_field_UP == e_UP)
    {
    int    index             = bennett_spins[bennett_index].Gamma_index_HS_field_UP;
    double exp_minus_delta_V = bennett_spins[bennett_index].exp_minus_delta_V_HS_field_UP;

    ratio_HS_field_UP = ctaux_tools.apply_bennett_on_Gamma(index, Gamma_up_size, Gamma_up_CPU,
    exp_minus_delta_V);
    }
    else
    {
    int    index             = bennett_spins[bennett_index].Gamma_index_HS_field_UP;
    double exp_minus_delta_V = bennett_spins[bennett_index].exp_minus_delta_V_HS_field_UP;

    ratio_HS_field_UP = ctaux_tools.apply_bennett_on_Gamma(index, Gamma_dn_size, Gamma_dn_CPU,
    exp_minus_delta_V);
    }

    assert(bennett_spins[bennett_index].HS_current_move==ANNIHILATION);

    double determinant_ratio = ratio_HS_field_UP*ratio_HS_field_DN;
    double acceptance_ratio  = calculate_acceptace_ratio(determinant_ratio,
    bennett_spins[bennett_index].HS_current_move);

    if( std::fabs(acceptance_ratio) >= rng.get_random_number() )
    {
    number_of_interacting_spins -= 1;

    if(acceptance_ratio < 0)
    sign *= -1;

    {// set column and row to zero
    if(bennett_spins[bennett_index].e_spin_HS_field_DN == e_UP)
    {
    int k = bennett_spins[bennett_index].Gamma_index_HS_field_DN;
    CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_up_CPU, k);
    }
    else
    {
    int k = bennett_spins[bennett_index].Gamma_index_HS_field_DN;
    CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_dn_CPU, k);
    }

    if(bennett_spins[bennett_index].e_spin_HS_field_UP == e_UP)
    {
    int k = bennett_spins[bennett_index].Gamma_index_HS_field_UP;
    CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_up_CPU, k);
    }
    else
    {
    int k = bennett_spins[bennett_index].Gamma_index_HS_field_UP;
    CT_AUX_WALKER_TOOLS<LIN_ALG::CPU>::set_to_identity(Gamma_dn_CPU, k);
    }
    }

    {// update the delayed spins and erase the bennett-spin
    int delayed_spin_index = bennett_spins[bennett_index].delayed_spin_index;

    assert(not delayed_spins[delayed_spin_index].is_a_bennett_spin);
    delayed_spins[delayed_spin_index].is_a_bennett_spin = true;

    bennett_spins.erase(bennett_spins.begin()+bennett_index, bennett_spins.begin()+bennett_index+1);
    }
    }
    else
    {
    Gamma_up_CPU.swap(stored_Gamma_up_CPU);
    Gamma_dn_CPU.swap(stored_Gamma_dn_CPU);
    }
  */
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::update_N_matrix_with_Gamma_matrix() {
  profiler_type profiler(__FUNCTION__, "CT-AUX walker", __LINE__, thread_id);

  // kills Bennett-spins and puts the interacting vertices all in the left part of the configuration
  SHRINK_TOOLS<device_t>::shrink_Gamma(configuration, Gamma_up, Gamma_dn);

  N_tools_obj.rebuild_N_matrix_via_Gamma_LU(configuration, N_up, Gamma_up, G_up, e_UP);
  N_tools_obj.rebuild_N_matrix_via_Gamma_LU(configuration, N_dn, Gamma_dn, G_dn, e_DN);

  configuration.commit_accepted_spins();
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
void MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::clean_up_the_configuration() {
  profiler_type profiler(__FUNCTION__, "CT-AUX walker", __LINE__, thread_id);

  SHRINK_tools_obj.reorganize_configuration_test(configuration, N_up, N_dn, G0_up, G0_dn);

  assert(configuration.assert_consistency());
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
HS_vertex_move_type MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::get_new_HS_move() {
  if (rng.get_random_number() > 0.50000000) {
    return CREATION;
  }

  if (configuration.get_number_of_interacting_HS_spins() == 0)
    return STATIC;

  return ANNIHILATION;
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
int MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::get_new_vertex_index(
    HS_vertex_move_type HS_current_move) {
  if (HS_current_move == CREATION)
    return configuration.get_random_noninteracting_vertex();

  return configuration.get_random_interacting_vertex();
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
HS_spin_states_type MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::get_new_spin_value(
    HS_vertex_move_type HS_current_move) {
  if (HS_current_move == CREATION) {
    if (rng.get_random_number() > 0.50000000)
      return HS_UP;
    else
      return HS_DN;
  }

  return HS_ZERO;
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
double MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::calculate_acceptace_ratio(
    double determinant_ratio, HS_vertex_move_type HS_current_move, double QMC_factor) {
  double N = number_of_interacting_spins;
  double K = parameters.get_K_CT_AUX();

  double acceptance_ratio;

  if (HS_current_move == CREATION) {
    acceptance_ratio = K / (N + 1.) * determinant_ratio / QMC_factor;
  }
  else {
    acceptance_ratio = (N / K) * determinant_ratio / QMC_factor;
  }

  return acceptance_ratio;
}

template <LIN_ALG::device_type device_t, class parameters_type, class MOMS_type>
bool MC_walker<CT_AUX_SOLVER, device_t, parameters_type, MOMS_type>::assert_exp_delta_V_value(
    HS_field_sign HS_field, int random_vertex_ind, HS_spin_states_type new_HS_spin_value,
    double exp_delta_V) {
  switch (HS_field) {
    case HS_FIELD_DN: {
      e_spin_states e_spin_HS_field_DN = configuration[random_vertex_ind].get_e_spins().first;
      int configuration_e_spin_index_HS_field_DN =
          configuration[random_vertex_ind].get_configuration_e_spin_indices().first;

      vertex_singleton_type& v_j =
          configuration.get(e_spin_HS_field_DN)[configuration_e_spin_index_HS_field_DN];

      if (std::fabs(CV_obj.exp_delta_V(v_j, new_HS_spin_value) - exp_delta_V) > 1.e-6) {
        std::cout << HS_field << "\t" << e_spin_HS_field_DN << std::endl;
        throw std::logic_error(__FUNCTION__);
      }
    } break;

    case HS_FIELD_UP: {
      e_spin_states e_spin_HS_field_UP = configuration[random_vertex_ind].get_e_spins().second;
      int configuration_e_spin_index_HS_field_UP =
          configuration[random_vertex_ind].get_configuration_e_spin_indices().second;

      vertex_singleton_type& v_j =
          configuration.get(e_spin_HS_field_UP)[configuration_e_spin_index_HS_field_UP];

      if (std::fabs(CV_obj.exp_delta_V(v_j, new_HS_spin_value) - exp_delta_V) > 1.e-6) {
        std::cout << HS_field << "\t" << e_spin_HS_field_UP << std::endl;
        throw std::logic_error(__FUNCTION__);
      }
    } break;

    default:
      throw std::logic_error(__FUNCTION__);
  };

  return true;
}
}
}

#endif
