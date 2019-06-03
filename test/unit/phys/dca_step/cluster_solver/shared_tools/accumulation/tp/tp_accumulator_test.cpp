// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file implements a no-change test for the two point accumulation of a mock configuration.

#include "dca/phys/dca_step/cluster_solver/shared_tools/accumulation/tp/tp_accumulator.hpp"

#include <array>
#include <map>
#include <string>
#include "gtest/gtest.h"

#include "dca/function/util/difference.hpp"
#include "dca/math/random/std_random_wrapper.hpp"
#include "dca/phys/four_point_type.hpp"
#include "test/unit/phys/dca_step/cluster_solver/shared_tools/accumulation/accumulation_test.hpp"
#include "test/unit/phys/dca_step/cluster_solver/test_setup.hpp"

constexpr bool update_baseline = false;

#define INPUT_DIR \
  DCA_SOURCE_DIR "/test/unit/phys/dca_step/cluster_solver/shared_tools/accumulation/tp/"

constexpr char input_file[] = INPUT_DIR "input_4x4.json";

using ConfigGenerator = dca::testing::AccumulationTest<double>;
using Configuration = ConfigGenerator::Configuration;
using Sample = ConfigGenerator::Sample;

using TpAccumulatorTest =
    dca::testing::G0Setup<dca::testing::LatticeBilayer, dca::phys::solver::CT_AUX, input_file>;

TEST_F(TpAccumulatorTest, Accumulate) {
  const std::array<int, 2> n{18, 22};
  Sample M;
  Configuration config;
  ConfigGenerator::prepareConfiguration(config, M, TpAccumulatorTest::BDmn::dmn_size(),
                                        TpAccumulatorTest::RDmn::dmn_size(), parameters_.get_beta(),
                                        n);

  Data::TpGreensFunction G4_check("G4");

  dca::io::HDF5Writer writer;
  dca::io::HDF5Reader reader;

  const std::string baseline = INPUT_DIR "tp_accumulator_test_baseline.hdf5";

  if (update_baseline)
    writer.open_file(baseline);
  else
    reader.open_file(baseline);

  std::map<dca::phys::FourPointType, std::string> func_names;
  func_names[dca::phys::PARTICLE_HOLE_TRANSVERSE] = "G4_ph_transverse";
  func_names[dca::phys::PARTICLE_HOLE_MAGNETIC] = "G4_ph_magnetic";
  func_names[dca::phys::PARTICLE_HOLE_CHARGE] = "G4_ph_charge";
  func_names[dca::phys::PARTICLE_PARTICLE_UP_DOWN] = "G4_pp_up_down";

  for (const dca::phys::FourPointType type :
       {dca::phys::PARTICLE_HOLE_TRANSVERSE, dca::phys::PARTICLE_HOLE_MAGNETIC,
        dca::phys::PARTICLE_HOLE_CHARGE, dca::phys::PARTICLE_PARTICLE_UP_DOWN}) {
    parameters_.set_four_point_type(type);

    dca::phys::solver::accumulator::TpAccumulator<Parameters> accumulator(
        data_->G0_k_w_cluster_excluded, parameters_);

    const int sign = 1;
    accumulator.accumulate(M, config, sign);
    accumulator.finalize();

    const auto& G4 = accumulator.get_sign_times_G4();

    if (update_baseline) {
      writer.execute(func_names[type], G4);
    }
    else {
      G4_check.set_name(func_names[type]);
      reader.execute(G4_check);
      const auto diff = dca::func::util::difference(G4, G4_check);
      EXPECT_GT(5e-7, diff.l_inf);
    }
  }

  if (update_baseline)
    writer.close_file();
  else
    reader.close_file();
}

using TpAccumulatorG4SumTest =
    dca::testing::G0Setup<dca::testing::LatticeBilayer, dca::phys::solver::CT_AUX, input_file>;

TEST_F(TpAccumulatorG4SumTest, Accumulate) {
  const std::array<int, 2> n{18, 22};
  Sample M;
  Configuration config;
  ConfigGenerator::prepareConfiguration(config, M, TpAccumulatorTest::BDmn::dmn_size(),
                                        TpAccumulatorTest::RDmn::dmn_size(), parameters_.get_beta(),
                                        n);

  Data::TpGreensFunctionSum G4_check_sum("G4_sum");
  
  dca::io::HDF5Writer writer;
  dca::io::HDF5Reader reader;

  const std::string baseline = INPUT_DIR "tp_accumulator_sum_test_baseline.hdf5";

  if (update_baseline)
    writer.open_file(baseline);
  else
    reader.open_file(baseline);

  std::map<dca::phys::FourPointType, std::string> func_names;
  func_names[dca::phys::PARTICLE_HOLE_TRANSVERSE] = "G4_ph_transverse";
  func_names[dca::phys::PARTICLE_HOLE_MAGNETIC] = "G4_ph_magnetic";
  func_names[dca::phys::PARTICLE_HOLE_CHARGE] = "G4_ph_charge";
  func_names[dca::phys::PARTICLE_PARTICLE_UP_DOWN] = "G4_pp_up_down";

  for (const dca::phys::FourPointType type :
       {dca::phys::PARTICLE_HOLE_TRANSVERSE, dca::phys::PARTICLE_HOLE_MAGNETIC,
        dca::phys::PARTICLE_HOLE_CHARGE, dca::phys::PARTICLE_PARTICLE_UP_DOWN}) {
    parameters_.set_four_point_type(type);
    parameters_.set_compute_all_frequency_transfers(true);


    dca::phys::solver::accumulator::TpAccumulator<Parameters> accumulator(
        data_->G0_k_w_cluster_excluded, parameters_);
    
    const int sign = 1;
    accumulator.accumulate(M, config, sign);
    accumulator.finalize();

    const auto& G4 = accumulator.get_sign_times_G4();

    if (update_baseline) {
      writer.execute(func_names[type], G4);
    }
    else {
      G4_check_sum.set_name(func_names[type]);
      reader.execute(G4_check_sum);
      const auto diff = dca::func::util::difference(G4, G4_check_sum);
      EXPECT_GT(5e-7, diff.l_inf);
    }
  }

  if (update_baseline)
    writer.close_file();
  else
    reader.close_file();
}
