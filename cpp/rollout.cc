// C++ rollout (ticket 13): loads the scene Python compiled (MuJoCo binary model, same 4.0.1 release), replays a
// scripted joint-target sequence per seed with the same reset, clipping, 20 Hz decimation and success rule as
// teleop_il.env.Env, and writes one result line per seed. The parity script compares it with the Python run.
//
// usage: rollout SCENE.mjb SEQUENCES.txt RESULTS.txt [TRACE.txt]
// SEQUENCES.txt: header lines "home h0..h5", "cube_half c", "lift_threshold l", "hold_steps n", "substeps s",
//   "episodes N", then per episode "episode seed cube_x cube_y steps" followed by `steps` lines of 6 joint targets.
// RESULTS.txt: per episode "seed success cube_z lift_height" (success 0 or 1).
// TRACE.txt (optional 4th argument): per control step "seed step qpos..." after the step, for the
//   trajectory comparison (end states alone hide physics changes: position control converges).
#include "sim.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>



int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: rollout SCENE.mjb SEQUENCES.txt RESULTS.txt [TRACE.txt]\n";
    return 2;
  }
  mjModel* m = mj_loadModel(argv[1], nullptr);
  if (!m) {
    std::cerr << "cannot load model " << argv[1] << "\n";
    return 2;
  }
  try {
    std::ifstream in(argv[2]);
    if (!in) throw std::runtime_error(std::string("cannot read ") + argv[2]);
    sim::Config c;
    sim::expect(in, "home");
    for (double& h : c.home) in >> h;
    sim::expect(in, "cube_half"); in >> c.cube_half;
    sim::expect(in, "lift_threshold"); in >> c.lift_threshold;
    sim::expect(in, "hold_steps"); in >> c.hold_steps;
    sim::expect(in, "substeps"); in >> c.substeps;
    sim::expect(in, "episodes"); in >> c.episodes;
    std::ofstream out(argv[3]);
    if (!out) throw std::runtime_error(std::string("cannot write ") + argv[3]);
    std::ofstream trace;
    if (argc == 5) {
      trace.open(argv[4]);
      if (!trace) throw std::runtime_error(std::string("cannot write ") + argv[4]);
    }
    sim::Rollout r(m, c);
    for (int e = 0; e < c.episodes; ++e) {
      long seed; double x, y; int steps;
      sim::expect(in, "episode");
      in >> seed >> x >> y >> steps;
      if (!in) throw std::runtime_error("bad episode header");
      r.reset(x, y);
      std::array<double, 6> a{};
      for (int t = 0; t < steps; ++t) {
        for (double& v : a) in >> v;
        if (!in) throw std::runtime_error("truncated action sequence");
        r.step(a.data());
        if (trace.is_open()) {
          trace << seed << ' ' << t;
          char num[32];
          for (int k = 0; k < m->nq; ++k) {
            std::snprintf(num, sizeof num, " %.17g", r.data()->qpos[k]);
            trace << num;
          }
          trace << '\n';
        }
      }
      char line[200];
      std::snprintf(line, sizeof line, "%ld %d %.17g %.17g\n", seed, r.succeeded() ? 1 : 0, r.cube_z(), r.lift_height());
      out << line;
    }
  } catch (const std::exception& ex) {
    std::cerr << "rollout: " << ex.what() << "\n";
    mj_deleteModel(m);
    return 1;
  }
  mj_deleteModel(m);
  return 0;
}
