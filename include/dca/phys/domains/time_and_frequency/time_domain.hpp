// Copyright (C) 2009-2016 ETH Zurich
// Copyright (C) 2007?-2016 Center for Nanophase Materials Sciences, ORNL
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Peter Staar (taa@zurich.ibm.com)
//         Urs R. Haehner (haehneru@itp.phys.ethz.ch)
//
// Time domain.

#ifndef DCA_PHYS_DOMAINS_TIME_AND_FREQUENCY_TIME_DOMAIN_HPP
#define DCA_PHYS_DOMAINS_TIME_AND_FREQUENCY_TIME_DOMAIN_HPP

#include <string>
#include <vector>

#include "dca/math/function_transform/domain_specifications.hpp"

namespace dca {
namespace phys {
namespace domains {
// dca::phys::domains::

class time_domain {
public:
  const static int RULE = 1;
  const static int DIMENSION = 1;

  typedef double scalar_type;
  typedef double element_type;

  typedef math::transform::interval_dmn_1D_type dmn_specifications_type;

  typedef time_domain parameter_type;  // Used in the interpolation.

  static int& get_size() {
    static int size = -1;
    return size;
  }

  static std::string get_name() {
    static std::string name = "time-domain";
    return name;
  }

  static std::vector<double>& get_elements() {
    static std::vector<double> elements;
    return elements;
  }

  static double get_beta() {
    return (get_elements().back() + 1.e-10);
  }

  static double get_volume() {
    return (get_elements().back() + 1.e-10);
  }

  template <typename Writer>
  static void write(Writer& writer);

  template <class stream_type>
  static void to_JSN(stream_type& ss);

  static void initialize(const double beta, const int time_slices, const double eps = 1.e-10);

  template <typename parameters_t>
  static void initialize(const parameters_t& parameters) {
    initialize(parameters.get_beta(), parameters.get_sp_time_intervals());
  }

  static void initialize_integration_domain(int level, std::vector<scalar_type>& weights,
                                            std::vector<element_type>& elements);
};

template <typename Writer>
void time_domain::write(Writer& writer) {
  writer.open_group(get_name());
  writer.execute("elements", get_elements());
  writer.close_group();
}

template <class stream_type>
void time_domain::to_JSN(stream_type& ss) {
  ss << "\"time_domain\" : [\n";

  for (int i = 0; i < get_size(); i++)
    if (i == get_size() - 1)
      ss << get_elements()[i] << "\n";
    else
      ss << get_elements()[i] << ",\n";

  ss << "]\n";
}

}  // domains
}  // phys
}  // dca

#endif  // DCA_PHYS_DOMAINS_TIME_AND_FREQUENCY_TIME_DOMAIN_HPP
