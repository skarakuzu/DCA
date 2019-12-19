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

#ifndef SINGLETON_OBJ_DEV_HPP
#define SINGLETON_OBJ_DEV_HPP

#ifndef DCA_HAVE_CUDA
#pragma error "This file requires CUDA."
#endif

#include <cuda.h>

namespace dca {
namespace phys {
namespace solver {
namespace ctaux {
// dca::phys::solver::ctaux::


  struct singleton_operator_dev {
    int b_ind;
    int r_ind;
    int t_ind;

    double t_val;
  };


}  // ctaux
}  // solver
}  // phys
}  // dca

#endif  // SINGLETON_OBJ_DEV_HPP
