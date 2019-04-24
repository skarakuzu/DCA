// Copyright (C) 2019 ETH Zurich
// Copyright (C) 2019 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Thomas A. Maier (maierta@ornl.gov)
//
// Template specialization of model parameters for square lattice two-orbital Hubbard model.

template <typename PointGroup>
class ModelParameters<models::TightBindingModel<models::twoorbital_lattice<PointGroup>>> {
public:
  ModelParameters() : t00_(0.), tp00_(0.), tpp00_(0.), t11_(0.), tp11_(0.), tpp11_(0.), t01_(0.), tp01_(0.), tpp01_(0.), DeltaE_(0.0), U_(0.), V_(0.) {}

  template <typename Concurrency>
  int getBufferSize(const Concurrency& concurrency) const;
  template <typename Concurrency>
  void pack(const Concurrency& concurrency, char* buffer, int buffer_size, int& position) const;
  template <typename Concurrency>
  void unpack(const Concurrency& concurrency, char* buffer, int buffer_size, int& position);

  template <typename ReaderOrWriter>
  void readWrite(ReaderOrWriter& reader_or_writer);

  double get_t00_() const {
    return t00_;
  }
  void set_t00(const double t00) {
    t00_ = t00;
  }

  double get_tp00_() const {
    return tp00_;
  }
  void set_tp00(const double tp00) {
    tp00_ = tp00;
  }

  double get_tpp00_() const {
    return tpp00_;
  }
  void set_tpp00(const double tpp00) {
    tpp00_ = tpp00;
  }

  double get_t11_() const {
    return t11_;
  }
  void set_t11(const double t11) {
    t11_ = t11;
  }

  double get_tp11_() const {
    return tp11_;
  }
  void set_tp11(const double tp11) {
    tp11_ = tp11;
  }

  double get_tpp11_() const {
    return tpp11_;
  }
  void set_tpp11(const double tpp11) {
    tpp11_ = tpp11;
  }

  double get_t01_() const {
    return t01_;
  }
  void set_t01(const double t01) {
    t01_ = t01;
  }

  double get_tp01_() const {
    return tp01_;
  }
  void set_tp01(const double tp01) {
    tp01_ = tp01;
  }

  double get_tpp01_() const {
    return tpp01_;
  }
  void set_tpp01(const double tpp01) {
    tpp01_ = tpp01;
  }

  double get_DeltaE_() const {
    return DeltaE_;
  }
  void set_DeltaE(const double DeltaE) {
    DeltaE_ = DeltaE;
  }

  double get_U() const {
    return U_;
  }
  void set_U(const double U) {
    U_ = U;
  }

  double get_V() const {
    return V_;
  }
  void set_V(const double V) {
    V_ = V;
  }


private:
  double t00_;
  double tp00_;
  double tpp00_;
  double t11_;
  double tp11_;
  double tpp11_;
  double t01_;
  double tp01_;
  double tpp01_;
  double DeltaE_;
  double U_;
  double V_;
};

template <typename PointGroup>
template <typename Concurrency>
int ModelParameters<models::TightBindingModel<models::twoorbital_lattice<PointGroup>>>::getBufferSize(
    const Concurrency& concurrency) const {
  int buffer_size = 0;

  buffer_size += concurrency.get_buffer_size(t00_);
  buffer_size += concurrency.get_buffer_size(tp00_);
  buffer_size += concurrency.get_buffer_size(tpp00_);
  buffer_size += concurrency.get_buffer_size(t11_);
  buffer_size += concurrency.get_buffer_size(tp11_);
  buffer_size += concurrency.get_buffer_size(tpp11_);
  buffer_size += concurrency.get_buffer_size(t01_);
  buffer_size += concurrency.get_buffer_size(tp01_);
  buffer_size += concurrency.get_buffer_size(tpp01_);
  buffer_size += concurrency.get_buffer_size(DeltaE_);
  buffer_size += concurrency.get_buffer_size(U_);
  buffer_size += concurrency.get_buffer_size(V_);

  return buffer_size;
}

template <typename PointGroup>
template <typename Concurrency>
void ModelParameters<models::TightBindingModel<models::twoorbital_lattice<PointGroup>>>::pack(
    const Concurrency& concurrency, char* buffer, int buffer_size, int& position) const {
  concurrency.pack(buffer, buffer_size, position, t00_);
  concurrency.pack(buffer, buffer_size, position, tp00_);
  concurrency.pack(buffer, buffer_size, position, tpp00_);
  concurrency.pack(buffer, buffer_size, position, t11_);
  concurrency.pack(buffer, buffer_size, position, tp11_);
  concurrency.pack(buffer, buffer_size, position, tpp11_);
  concurrency.pack(buffer, buffer_size, position, t01_);
  concurrency.pack(buffer, buffer_size, position, tp01_);
  concurrency.pack(buffer, buffer_size, position, tpp01_);
  concurrency.pack(buffer, buffer_size, position, DeltaE_);
  concurrency.pack(buffer, buffer_size, position, U_);
  concurrency.pack(buffer, buffer_size, position, V_);
}

template <typename PointGroup>
template <typename Concurrency>
void ModelParameters<models::TightBindingModel<models::twoorbital_lattice<PointGroup>>>::unpack(
    const Concurrency& concurrency, char* buffer, int buffer_size, int& position) {
  concurrency.unpack(buffer, buffer_size, position, t00_);
  concurrency.unpack(buffer, buffer_size, position, tp00_);
  concurrency.unpack(buffer, buffer_size, position, tpp00_);
  concurrency.unpack(buffer, buffer_size, position, t11_);
  concurrency.unpack(buffer, buffer_size, position, tp11_);
  concurrency.unpack(buffer, buffer_size, position, tpp11_);
  concurrency.unpack(buffer, buffer_size, position, t01_);
  concurrency.unpack(buffer, buffer_size, position, tp01_);
  concurrency.unpack(buffer, buffer_size, position, tpp01_);
  concurrency.unpack(buffer, buffer_size, position, DeltaE_);
  concurrency.unpack(buffer, buffer_size, position, U_);
  concurrency.unpack(buffer, buffer_size, position, V_);
}

template <typename PointGroup>
template <typename ReaderOrWriter>
void ModelParameters<models::TightBindingModel<models::twoorbital_lattice<PointGroup>>>::readWrite(
    ReaderOrWriter& reader_or_writer) {
  try {
    reader_or_writer.open_group("twoorbital-Hubbard-model");

    try {
      reader_or_writer.execute("t00", t00_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("tp00", tp00_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("tpp00", tpp00_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("t11", t11_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("tp11", tp11_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("tpp11", tpp11_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("t01", t01_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("tp01", tp01_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("tpp01", tpp01_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("DeltaE", DeltaE_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("U", U_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("V", V_);
    }
    catch (const std::exception& r_e) {
    }

    reader_or_writer.close_group();
  }
  catch (const std::exception& r_e) {
  }
}
