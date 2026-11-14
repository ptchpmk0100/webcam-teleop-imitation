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
#include <mujoco/mujoco.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* kArmJoints[6] = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};

struct Config {
  std::array<double, 6> home{};
  double cube_half = 0, lift_threshold = 0;
  int hold_steps = 0, substeps = 0, episodes = 0;
};

std::string expect(std::istream& in, const std::string& key) {
  std::string word;
  if (!(in >> word) || word != key) throw std::runtime_error("expected '" + key + "', got '" + word + "'");
  return word;
}

class Rollout {
 public:
  Rollout(mjModel* m, const Config& c) : m_(m), c_(c), d_(mj_makeData(m)) {
    for (int i = 0; i < 6; ++i) {
      int j = mj_name2id(m, mjOBJ_JOINT, kArmJoints[i]);
      if (j < 0) throw std::runtime_error(std::string("joint not found: ") + kArmJoints[i]);
      arm_qadr_[i] = m->jnt_qposadr[j];
    }
    int cube = mj_name2id(m, mjOBJ_BODY, "cube");
    if (cube < 0) throw std::runtime_error("body 'cube' not found");
    cube_qadr_ = m->jnt_qposadr[m->body_jntadr[cube]];
  }
  ~Rollout() { mj_deleteData(d_); }

  void reset(double x, double y) {  // mirrors Env.reset after its random draw
    mj_resetData(m_, d_);
    for (int i = 0; i < 6; ++i) {
      d_->qpos[arm_qadr_[i]] = c_.home[i];
      d_->ctrl[i] = c_.home[i];
    }
    const double cube[7] = {x, y, c_.cube_half, 1, 0, 0, 0};
    for (int k = 0; k < 7; ++k) d_->qpos[cube_qadr_ + k] = cube[k];
    mj_forward(m_, d_);
    hold_ = 0;
    succeeded_ = false;
  }

  void step(const double* action) {  // mirrors Env.step: clip, then `substeps` physics steps, hold counter per step
    for (int i = 0; i < m_->nu; ++i) {
      const double lo = m_->actuator_ctrlrange[2 * i], hi = m_->actuator_ctrlrange[2 * i + 1];
      d_->ctrl[i] = std::fmin(std::fmax(action[i], lo), hi);
    }
    for (int s = 0; s < c_.substeps; ++s) mj_step(m_, d_);
    hold_ = lift_height() > c_.lift_threshold ? hold_ + 1 : 0;
    if (hold_ >= c_.hold_steps) succeeded_ = true;
  }

  double lift_height() const {  // lowest point of the cube above the table, like Env.lift_height
    mjtNum rot[9];
    mju_quat2Mat(rot, d_->qpos + cube_qadr_ + 3);
    const double half_extent_z = c_.cube_half * (std::fabs(rot[6]) + std::fabs(rot[7]) + std::fabs(rot[8]));
    return d_->qpos[cube_qadr_ + 2] - half_extent_z;
  }
  double cube_z() const { return d_->qpos[cube_qadr_ + 2]; }
  const mjData* data() const { return d_; }
  bool succeeded() const { return succeeded_; }

 private:
  mjModel* m_;
  Config c_;
  mjData* d_;
  std::array<int, 6> arm_qadr_{};
  int cube_qadr_ = 0, hold_ = 0;
  bool succeeded_ = false;
};

}  // namespace

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
    Config c;
    expect(in, "home");
    for (double& h : c.home) in >> h;
    expect(in, "cube_half"); in >> c.cube_half;
    expect(in, "lift_threshold"); in >> c.lift_threshold;
    expect(in, "hold_steps"); in >> c.hold_steps;
    expect(in, "substeps"); in >> c.substeps;
    expect(in, "episodes"); in >> c.episodes;
    std::ofstream out(argv[3]);
    if (!out) throw std::runtime_error(std::string("cannot write ") + argv[3]);
    std::ofstream trace;
    if (argc == 5) {
      trace.open(argv[4]);
      if (!trace) throw std::runtime_error(std::string("cannot write ") + argv[4]);
    }
    Rollout r(m, c);
    for (int e = 0; e < c.episodes; ++e) {
      long seed; double x, y; int steps;
      expect(in, "episode");
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
