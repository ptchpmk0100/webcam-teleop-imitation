// C++ rollout of state policy v2 (ticket 24): the same cube-relative observation, ONNX network (contract v2), chunk
// blending, open/closed hysteresis and gripper-move controller as teleop_il.policy_v2.PolicyV2 and
// teleop_il.controller.GripperController, in the same simulation as Python (sim.h).
//
// usage:
//   policy_v2_rollout infer MODEL.onnx OBS.txt OUT.txt
//       OBS.txt: one observation per line (6 numbers); OUT.txt: the chunk per line (chunk x 4 numbers).
//   policy_v2_rollout controller CONTROLLER.mjb PARAMS.txt CASES.txt OUT.txt
//       CASES.txt: per line 6 joint values, a target x y z and closed (0/1); OUT.txt: per line the 6 joint targets
//       and the gripper position of the given joints (forward kinematics).
//   policy_v2_rollout rollout SCENE.mjb CONTROLLER.mjb POLICY PARAMS.txt EPISODES.txt RESULTS.txt
//       POLICY: a contract-v2 .onnx file, "hold" (keep the starting joints) or "random:SEED".
//       EPISODES.txt as for policy_rollout (ticket 14); RESULTS.txt: per episode "seed success cube_z lift steps".
// PARAMS.txt (written by teleop_il.cpp_policy_v2_parity from the Python constants): "iterations n",
// "orientation_weight w", "damping d", "gripper_open g", "gripper_closed g", "home h0..h4", "grasp_z z",
// "jaw_offset j", "cube_half c", "chunk h", "m m", "close_above p", "open_below p".
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "sim.h"

namespace {

constexpr int kObs = 6, kAct = 4, kArm = 5;

struct Params {
  int iterations = 0, chunk = 0;
  double orientation_weight = 0, damping = 0, gripper_open = 0, gripper_closed = 0, grasp_z = 0, jaw_offset = 0,
         cube_half = 0, m = 0, close_above = 0, open_below = 0;
  std::array<double, kArm> home{};
};

Params read_params(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  Params p;
  sim::expect(in, "iterations"); in >> p.iterations;
  sim::expect(in, "orientation_weight"); in >> p.orientation_weight;
  sim::expect(in, "damping"); in >> p.damping;
  sim::expect(in, "gripper_open"); in >> p.gripper_open;
  sim::expect(in, "gripper_closed"); in >> p.gripper_closed;
  sim::expect(in, "home");
  for (double& h : p.home) in >> h;
  sim::expect(in, "grasp_z"); in >> p.grasp_z;
  sim::expect(in, "jaw_offset"); in >> p.jaw_offset;
  sim::expect(in, "cube_half"); in >> p.cube_half;
  sim::expect(in, "chunk"); in >> p.chunk;
  sim::expect(in, "m"); in >> p.m;
  sim::expect(in, "close_above"); in >> p.close_above;
  sim::expect(in, "open_below"); in >> p.open_below;
  if (!in) throw std::runtime_error("bad parameter file " + path);
  return p;
}

using Mat3 = std::array<double, 9>;  // row-major, like MuJoCo's xmat

Mat3 matmul(const Mat3& a, const Mat3& b) {
  Mat3 c{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) c[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  return c;
}

Mat3 rot_z(double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  return {c, -s, 0, s, c, 0, 0, 0, 1};
}

// Smallest rotation that turns the gripper approach axis (frame x, the first column) to straight down.
Mat3 align_axis_down(const Mat3& r) {
  const double a[3] = {r[0], r[3], r[6]}, down[3] = {0, 0, -1};
  const double v[3] = {a[1] * down[2] - a[2] * down[1], a[2] * down[0] - a[0] * down[2], a[0] * down[1] - a[1] * down[0]};
  const double s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const double c = a[0] * down[0] + a[1] * down[1] + a[2] * down[2];
  if (s < 1e-12) return r;
  const Mat3 k{0, -v[2] / s, v[1] / s, v[2] / s, 0, -v[0] / s, -v[1] / s, v[0] / s, 0};
  const Mat3 kk = matmul(k, k);
  Mat3 e{};
  for (int i = 0; i < 9; ++i) e[i] = (i % 4 == 0 ? 1.0 : 0.0) + s * k[i] + (1 - c) * kk[i];
  return matmul(e, r);
}

// Solve A x = b (n x n, row-major) by Gaussian elimination with partial pivoting.
std::vector<double> solve(std::vector<double> a, std::vector<double> b, int n) {
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (std::fabs(a[r * n + col]) > std::fabs(a[piv * n + col])) piv = r;
    if (piv != col) {
      for (int k = 0; k < n; ++k) std::swap(a[col * n + k], a[piv * n + k]);
      std::swap(b[col], b[piv]);
    }
    for (int r = col + 1; r < n; ++r) {
      const double f = a[r * n + col] / a[col * n + col];
      for (int k = col; k < n; ++k) a[r * n + k] -= f * a[col * n + k];
      b[r] -= f * b[col];
    }
  }
  std::vector<double> x(n);
  for (int r = n - 1; r >= 0; --r) {
    double s = b[r];
    for (int k = r + 1; k < n; ++k) s -= a[r * n + k] * x[k];
    x[r] = s / a[r * n + r];
  }
  return x;
}

// teleop_il.controller.GripperController: damped least squares on the gripper site, from the measured joints.
class GripperController {
 public:
  GripperController(const mjModel* m, const Params& p) : m_(m), p_(p), d_(mj_makeData(m)) {
    for (int i = 0; i < kArm; ++i) {
      const int j = mj_name2id(m, mjOBJ_JOINT, sim::kArmJoints[i]);
      if (j < 0) throw std::runtime_error(std::string("joint not found: ") + sim::kArmJoints[i]);
      qadr_[i] = m->jnt_qposadr[j];
      dofadr_[i] = m->jnt_dofadr[j];
      lo_[i] = m->actuator_ctrlrange[2 * i];
      hi_[i] = m->actuator_ctrlrange[2 * i + 1];
    }
    site_ = mj_name2id(m, mjOBJ_SITE, "gripperframe");
    if (site_ < 0) throw std::runtime_error("site 'gripperframe' not found");
    jac_p_.resize(3 * m->nv);
    jac_r_.resize(3 * m->nv);
    const std::array<double, 3> home = kinematics(p.home.data());
    Mat3 r;
    std::copy(d_->site_xmat + 9 * site_, d_->site_xmat + 9 * site_ + 9, r.begin());
    home_rotation_ = align_axis_down(r);
    home_yaw_ = std::atan2(home[1], home[0]);
  }
  ~GripperController() { mj_deleteData(d_); }
  GripperController(const GripperController&) = delete;
  GripperController& operator=(const GripperController&) = delete;

  std::array<double, 3> gripper_position(const double* joints) { return kinematics(joints); }

  // 6 joint targets that put the gripper frame at `target`, from the measured `joints` (the first 5 are used).
  std::array<double, 6> command(const double* joints, const double* target, bool closed) {
    std::array<double, kArm> q;
    for (int i = 0; i < kArm; ++i) q[i] = std::fmin(std::fmax(joints[i], lo_[i]), hi_[i]);
    const double yaw = std::atan2(target[1], target[0]) - home_yaw_;
    const Mat3 goal = matmul(rot_z(yaw), home_rotation_);
    mjtNum goal_quat[4], quat[4], quat_conj[4], err_quat[4], err_rot[3];
    mju_mat2Quat(goal_quat, goal.data());
    const double w = p_.orientation_weight;
    for (int it = 0; it < p_.iterations; ++it) {
      const std::array<double, 3> x = kinematics(q.data());
      mju_mat2Quat(quat, d_->site_xmat + 9 * site_);
      mju_negQuat(quat_conj, quat);
      mju_mulQuat(err_quat, goal_quat, quat_conj);
      mju_quat2Vel(err_rot, err_quat, 1.0);
      mj_jacSite(m_, d_, jac_p_.data(), jac_r_.data(), site_);
      std::array<double, 6 * kArm> jac;
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < kArm; ++c) {
          jac[r * kArm + c] = jac_p_[r * m_->nv + dofadr_[c]];
          jac[(3 + r) * kArm + c] = w * jac_r_[r * m_->nv + dofadr_[c]];
        }
      std::array<double, 6> err;
      for (int r = 0; r < 3; ++r) {
        err[r] = target[r] - x[r];
        err[3 + r] = w * err_rot[r];
      }
      q = limited_step(q, jac, err);
    }
    return {q[0], q[1], q[2], q[3], q[4], closed ? p_.gripper_closed : p_.gripper_open};
  }

 private:
  std::array<double, 3> kinematics(const double* arm) {
    for (int i = 0; i < kArm; ++i) d_->qpos[qadr_[i]] = arm[i];
    mj_kinematics(m_, d_);
    mj_comPos(m_, d_);
    return {d_->site_xpos[3 * site_], d_->site_xpos[3 * site_ + 1], d_->site_xpos[3 * site_ + 2]};
  }

  // One damped least-squares step; a joint the step would push past its limit is put on the limit and left out of a
  // re-solve, so the other joints take over its share.
  std::array<double, kArm> limited_step(const std::array<double, kArm>& q, const std::array<double, 6 * kArm>& jac,
                                        const std::array<double, 6>& err) {
    std::array<bool, kArm> free;
    free.fill(true);
    std::array<double, kArm> base = q, next = q;
    for (int round = 0; round < kArm; ++round) {
      std::array<double, 6> remaining;
      for (int r = 0; r < 6; ++r) {
        double s = 0;
        for (int c = 0; c < kArm; ++c) s += jac[r * kArm + c] * (base[c] - q[c]);
        remaining[r] = err[r] - s;
      }
      std::vector<int> idx;
      for (int c = 0; c < kArm; ++c)
        if (free[c]) idx.push_back(c);
      const int n = static_cast<int>(idx.size());
      std::vector<double> a(n * n), b(n);
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
          double s = 0;
          for (int r = 0; r < 6; ++r) s += jac[r * kArm + idx[i]] * jac[r * kArm + idx[k]];
          a[i * n + k] = s + (i == k ? p_.damping : 0.0);
        }
        double s = 0;
        for (int r = 0; r < 6; ++r) s += jac[r * kArm + idx[i]] * remaining[r];
        b[i] = s;
      }
      const std::vector<double> dq = solve(a, b, n);
      next = base;
      for (int i = 0; i < n; ++i) next[idx[i]] += dq[i];
      bool any_over = false;
      for (int c = 0; c < kArm; ++c) {
        if (free[c] && (next[c] < lo_[c] || next[c] > hi_[c])) {
          base[c] = std::fmin(std::fmax(next[c], lo_[c]), hi_[c]);
          free[c] = false;
          any_over = true;
        }
      }
      if (!any_over) return next;
      if (std::none_of(free.begin(), free.end(), [](bool f) { return f; })) return base;
    }
    for (int c = 0; c < kArm; ++c) next[c] = std::fmin(std::fmax(next[c], lo_[c]), hi_[c]);
    return next;
  }

  const mjModel* m_;
  Params p_;
  mjData* d_;
  std::array<int, kArm> qadr_{}, dofadr_{};
  std::array<double, kArm> lo_{}, hi_{};
  int site_ = -1;
  std::vector<mjtNum> jac_p_, jac_r_;
  Mat3 home_rotation_{};
  double home_yaw_ = 0;
};

class OnnxNetwork {
 public:
  OnnxNetwork(const std::string& path, int chunk)
      : env_(ORT_LOGGING_LEVEL_WARNING, "policy_v2_rollout"), session_(env_, path.c_str(), options()), chunk_(chunk) {}

  // (n, 6) float32 in, (n, chunk, 4) float32 out.
  std::vector<float> run(const std::vector<float>& obs, int64_t n) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> shape{n, kObs};
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(obs.data()), obs.size(),
                                                       shape.data(), shape.size());
    const char* in_names[] = {"observation"};
    const char* out_names[] = {"actions"};
    auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    const auto info = out[0].GetTensorTypeAndShapeInfo().GetShape();
    if (info.size() != 3 || info[1] != chunk_ || info[2] != kAct)
      throw std::runtime_error("the network's output is not (batch, chunk, 4)");
    const float* p = out[0].GetTensorData<float>();
    return std::vector<float>(p, p + n * chunk_ * kAct);
  }

 private:
  static Ort::SessionOptions options() {
    Ort::SessionOptions o;
    o.SetIntraOpNumThreads(1);
    return o;
  }
  Ort::Env env_;
  Ort::Session session_;
  int chunk_;
};

// numpy's float64 add.reduce (pairwise summation) for fewer than 128 values, so the blending weights' sum is the same.
double numpy_sum(const std::vector<double>& a) {
  const size_t n = a.size();
  if (n < 8) {
    double s = 0;
    for (double v : a) s += v;
    return s;
  }
  double r[8];
  for (int j = 0; j < 8; ++j) r[j] = a[j];
  size_t i = 8;
  for (; i < n - n % 8; i += 8)
    for (int j = 0; j < 8; ++j) r[j] += a[i + j];
  double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
  for (; i < n; ++i) s += a[i];
  return s;
}

// teleop_il.policy_v2.features: gripper minus the cube's grasp point, gripper height, cube bottom height, gripper joint.
std::array<float, kObs> features(const double* obs13, const std::array<double, 3>& ee, const Params& p) {
  const double cx = obs13[6], cy = obs13[7], cz = obs13[8];
  const double yaw = std::atan2(cy, cx);
  const double axis[2] = {-std::sin(yaw), std::cos(yaw)};
  const double goal[3] = {cx + (-p.jaw_offset * axis[0]), cy + (-p.jaw_offset * axis[1]), cz + p.grasp_z - p.cube_half};
  return {static_cast<float>(ee[0] - goal[0]), static_cast<float>(ee[1] - goal[1]), static_cast<float>(ee[2] - goal[2]),
          static_cast<float>(ee[2]), static_cast<float>(cz - p.cube_half), static_cast<float>(obs13[5])};
}

// teleop_il.policy_v2.PolicyV2 with execute="ensemble".
class PolicyV2 {
 public:
  PolicyV2(OnnxNetwork* net, GripperController* controller, const Params& p) : net_(net), c_(controller), p_(p) {}
  void reset() {
    t_ = 0;
    chunks_.clear();
    closed_ = false;
  }
  std::array<double, 6> operator()(const double* obs13) {
    const std::array<double, 3> ee = c_->gripper_position(obs13);
    const std::array<float, kObs> x = features(obs13, ee, p_);
    const std::vector<float> out = net_->run(std::vector<float>(x.begin(), x.end()), 1);
    Chunk ch{t_, std::vector<std::array<double, 3>>(p_.chunk), std::vector<double>(p_.chunk)};
    for (int k = 0; k < p_.chunk; ++k) {
      for (int r = 0; r < 3; ++r) ch.targets[k][r] = ee[r] + static_cast<double>(out[k * kAct + r]);
      ch.probs[k] = 1.0 / (1.0 + std::exp(-static_cast<double>(out[k * kAct + 3])));
    }
    chunks_.push_back(std::move(ch));
    while (t_ - chunks_.front().start >= p_.chunk) chunks_.pop_front();
    std::vector<double> w(chunks_.size());
    for (size_t i = 0; i < w.size(); ++i) w[i] = std::exp(-p_.m * static_cast<double>(i));   // i = 0: oldest chunk
    const double wsum = numpy_sum(w);
    double target[3] = {0, 0, 0}, prob = 0;
    for (size_t i = 0; i < chunks_.size(); ++i) {
      const int k = t_ - chunks_[i].start;
      for (int r = 0; r < 3; ++r) target[r] += w[i] * chunks_[i].targets[k][r];
      prob += w[i] * chunks_[i].probs[k];
    }
    for (double& v : target) v /= wsum;
    prob /= wsum;
    if (closed_ && prob < p_.open_below) closed_ = false;
    else if (!closed_ && prob > p_.close_above) closed_ = true;
    ++t_;
    return c_->command(obs13, target, closed_);
  }

 private:
  struct Chunk {
    int start;
    std::vector<std::array<double, 3>> targets;
    std::vector<double> probs;
  };
  OnnxNetwork* net_;
  GripperController* c_;
  Params p_;
  int t_ = 0;
  std::deque<Chunk> chunks_;
  bool closed_ = false;
};

std::vector<double> read_numbers(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::vector<double> v;
  double x;
  while (in >> x) v.push_back(x);
  return v;
}

void write_row(std::ofstream& out, const double* v, int n) {
  char num[32];
  for (int k = 0; k < n; ++k) {
    std::snprintf(num, sizeof num, k ? " %.17g" : "%.17g", v[k]);
    out << num;
  }
  out << '\n';
}

using ModelPtr = std::unique_ptr<mjModel, void (*)(mjModel*)>;

ModelPtr load(const std::string& path) {
  mjModel* m = mj_loadModel(path.c_str(), nullptr);
  if (!m) throw std::runtime_error("cannot load model " + path);
  return ModelPtr(m, mj_deleteModel);
}

int infer(const std::string& model, const std::string& obs_path, const std::string& out_path) {
  const std::vector<double> v = read_numbers(obs_path);
  if (v.size() % kObs != 0) throw std::runtime_error("observation file is not a multiple of 6 numbers");
  const int64_t n = v.size() / kObs;
  std::vector<float> obs(v.begin(), v.end());
  // the chunk length is read from the model itself
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "policy_v2_rollout");
  Ort::SessionOptions o;
  o.SetIntraOpNumThreads(1);
  Ort::Session probe(env, model.c_str(), o);
  const int chunk = static_cast<int>(probe.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1]);
  OnnxNetwork net(model, chunk);
  const std::vector<float> act = net.run(obs, n);
  std::ofstream out(out_path);
  std::vector<double> row(chunk * kAct);
  for (int64_t i = 0; i < n; ++i) {
    for (int k = 0; k < chunk * kAct; ++k) row[k] = act[i * chunk * kAct + k];
    write_row(out, row.data(), chunk * kAct);
  }
  return 0;
}

int controller(const std::string& controller_mjb, const std::string& params, const std::string& cases,
               const std::string& out_path) {
  ModelPtr m = load(controller_mjb);
  const Params p = read_params(params);
  GripperController c(m.get(), p);
  const std::vector<double> v = read_numbers(cases);
  if (v.size() % 10 != 0) throw std::runtime_error("case file is not a multiple of 10 numbers");
  std::ofstream out(out_path);
  for (size_t i = 0; i < v.size(); i += 10) {
    const std::array<double, 6> cmd = c.command(&v[i], &v[i + 6], v[i + 9] > 0.5);
    const std::array<double, 3> ee = c.gripper_position(&v[i]);
    double row[9];
    for (int k = 0; k < 6; ++k) row[k] = cmd[k];
    for (int k = 0; k < 3; ++k) row[6 + k] = ee[k];
    write_row(out, row, 9);
  }
  return 0;
}

int rollout(const std::string& scene, const std::string& controller_mjb, const std::string& policy_name,
            const std::string& params, const std::string& episodes_path, const std::string& results_path) {
  ModelPtr m = load(scene), cm = load(controller_mjb);
  const Params p = read_params(params);
  std::ifstream in(episodes_path);
  if (!in) throw std::runtime_error("cannot read " + episodes_path);
  sim::Config c;
  int max_steps = 0;
  sim::expect(in, "home");
  for (double& h : c.home) in >> h;
  sim::expect(in, "cube_half"); in >> c.cube_half;
  sim::expect(in, "lift_threshold"); in >> c.lift_threshold;
  sim::expect(in, "hold_steps"); in >> c.hold_steps;
  sim::expect(in, "substeps"); in >> c.substeps;
  sim::expect(in, "max_steps"); in >> max_steps;
  sim::expect(in, "episodes"); in >> c.episodes;

  std::unique_ptr<OnnxNetwork> net;
  std::unique_ptr<GripperController> ctl;
  std::unique_ptr<PolicyV2> policy;
  std::mt19937_64 rng;
  const bool hold = policy_name == "hold";
  const bool random = policy_name.rfind("random:", 0) == 0;
  if (random) {
    rng.seed(std::stoull(policy_name.substr(7)));
  } else if (!hold) {
    net = std::make_unique<OnnxNetwork>(policy_name, p.chunk);
    ctl = std::make_unique<GripperController>(cm.get(), p);
    policy = std::make_unique<PolicyV2>(net.get(), ctl.get(), p);
  }

  sim::Rollout r(m.get(), c);
  std::ofstream out(results_path);
  if (!out) throw std::runtime_error("cannot write " + results_path);
  for (int e = 0; e < c.episodes; ++e) {
    long seed;
    double x, y;
    sim::expect(in, "episode");
    in >> seed >> x >> y;
    if (!in) throw std::runtime_error("bad episode line");
    r.reset(x, y);
    if (policy) policy->reset();
    double obs[13], start[6], action[6];
    r.observation(obs);
    for (int i = 0; i < 6; ++i) start[i] = obs[i];
    int steps = 0;
    for (; steps < max_steps; ++steps) {
      r.observation(obs);
      if (hold) {
        for (int i = 0; i < 6; ++i) action[i] = start[i];
      } else if (random) {
        for (int i = 0; i < 6; ++i) {
          std::uniform_real_distribution<double> u(m->actuator_ctrlrange[2 * i], m->actuator_ctrlrange[2 * i + 1]);
          action[i] = u(rng);
        }
      } else {
        const std::array<double, 6> a = (*policy)(obs);
        for (int i = 0; i < 6; ++i) action[i] = a[i];
      }
      r.step(action);
      if (r.succeeded()) { ++steps; break; }
    }
    char line[200];
    std::snprintf(line, sizeof line, "%ld %d %.17g %.17g %d\n", seed, r.succeeded() ? 1 : 0, r.cube_z(),
                  r.lift_height(), steps);
    out << line;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "infer" && argc == 5) return infer(argv[2], argv[3], argv[4]);
    if (mode == "controller" && argc == 6) return controller(argv[2], argv[3], argv[4], argv[5]);
    if (mode == "rollout" && argc == 8) return rollout(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
    std::cerr << "usage: policy_v2_rollout infer MODEL.onnx OBS.txt OUT.txt\n"
                 "       policy_v2_rollout controller CONTROLLER.mjb PARAMS.txt CASES.txt OUT.txt\n"
                 "       policy_v2_rollout rollout SCENE.mjb CONTROLLER.mjb (MODEL.onnx|hold|random:SEED) PARAMS.txt "
                 "EPISODES.txt RESULTS.txt\n";
    return 2;
  } catch (const std::exception& ex) {
    std::cerr << "policy_v2_rollout: " << ex.what() << "\n";
    return 1;
  }
}
