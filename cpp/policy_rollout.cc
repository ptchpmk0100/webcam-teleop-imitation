// C++ policy rollout (ticket 14): runs the exported ONNX policy with ONNX Runtime in the same simulation as Python.
//
// usage:
//   policy_rollout infer MODEL.onnx OBS.txt OUT.txt
//       OBS.txt: one observation per line (13 numbers); OUT.txt: the 10 x 6 action chunk per line (60 numbers).
//   policy_rollout rollout SCENE.mjb POLICY EPISODES.txt RESULTS.txt
//       POLICY: a .onnx file, "hold" (keep the starting joints) or "random:SEED" (uniform in the joint limits).
//       EPISODES.txt: "home h0..h5", "cube_half c", "lift_threshold l", "hold_steps n", "substeps s", "max_steps m",
//       "episodes N", then N lines "episode seed cube_x cube_y".
//       RESULTS.txt: per episode "seed success cube_z lift_height steps".
// The loop mirrors teleop_il.evaluation.evaluate: observe, act, step, stop on success or after max_steps; the ONNX
// policy is fed float32 state and its chunk's first action is executed, like teleop_il.policy.OnnxPolicy.
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sim.h"

namespace {

constexpr int kObs = 13, kChunk = 10, kAct = 6;

class OnnxPolicy {
 public:
  explicit OnnxPolicy(const std::string& path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "policy_rollout"), session_(env_, path.c_str(), options()) {}

  // (n, 13) float32 in, (n, 10, 6) float32 out.
  std::vector<float> run(const std::vector<float>& obs, int64_t n) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> shape{n, kObs};
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(obs.data()), obs.size(),
                                                       shape.data(), shape.size());
    const char* in_names[] = {"observation"};
    const char* out_names[] = {"actions"};
    auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    const float* p = out[0].GetTensorData<float>();
    return std::vector<float>(p, p + n * kChunk * kAct);
  }

 private:
  static Ort::SessionOptions options() {
    Ort::SessionOptions o;
    o.SetIntraOpNumThreads(1);
    return o;
  }
  Ort::Env env_;
  Ort::Session session_;
};

int infer(const std::string& model, const std::string& obs_path, const std::string& out_path) {
  OnnxPolicy policy(model);
  std::ifstream in(obs_path);
  if (!in) throw std::runtime_error("cannot read " + obs_path);
  std::vector<float> obs;
  double v;
  while (in >> v) obs.push_back(static_cast<float>(v));
  if (obs.size() % kObs != 0) throw std::runtime_error("observation file is not a multiple of 13 numbers");
  const int64_t n = obs.size() / kObs;
  const std::vector<float> act = policy.run(obs, n);
  std::ofstream out(out_path);
  char num[32];
  for (int64_t i = 0; i < n; ++i) {
    for (int k = 0; k < kChunk * kAct; ++k) {
      std::snprintf(num, sizeof num, k ? " %.9g" : "%.9g", act[i * kChunk * kAct + k]);
      out << num;
    }
    out << '\n';
  }
  return 0;
}

int rollout(const std::string& scene, const std::string& policy_name, const std::string& episodes_path,
            const std::string& results_path) {
  mjModel* m = mj_loadModel(scene.c_str(), nullptr);
  if (!m) throw std::runtime_error("cannot load model " + scene);
  std::unique_ptr<mjModel, void (*)(mjModel*)> model_guard(m, mj_deleteModel);
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

  std::unique_ptr<OnnxPolicy> onnx;
  std::mt19937_64 rng;
  const bool hold = policy_name == "hold";
  const bool random = policy_name.rfind("random:", 0) == 0;
  if (random) rng.seed(std::stoull(policy_name.substr(7)));
  else if (!hold) onnx = std::make_unique<OnnxPolicy>(policy_name);

  sim::Rollout r(m, c);
  std::ofstream out(results_path);
  if (!out) throw std::runtime_error("cannot write " + results_path);
  for (int e = 0; e < c.episodes; ++e) {
    long seed; double x, y;
    sim::expect(in, "episode");
    in >> seed >> x >> y;
    if (!in) throw std::runtime_error("bad episode line");
    r.reset(x, y);
    double obs[kObs], start[kAct], action[kAct];
    r.observation(obs);
    for (int i = 0; i < kAct; ++i) start[i] = obs[i];
    int steps = 0;
    for (; steps < max_steps; ++steps) {
      r.observation(obs);
      if (hold) {
        for (int i = 0; i < kAct; ++i) action[i] = start[i];
      } else if (random) {
        for (int i = 0; i < kAct; ++i) {
          std::uniform_real_distribution<double> u(m->actuator_ctrlrange[2 * i], m->actuator_ctrlrange[2 * i + 1]);
          action[i] = u(rng);
        }
      } else {
        std::vector<float> o(obs, obs + kObs);
        const std::vector<float> chunk = onnx->run(o, 1);
        for (int i = 0; i < kAct; ++i) action[i] = chunk[i];       // the chunk's first action
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
    if (mode == "rollout" && argc == 6) return rollout(argv[2], argv[3], argv[4], argv[5]);
    std::cerr << "usage: policy_rollout infer MODEL.onnx OBS.txt OUT.txt\n"
                 "       policy_rollout rollout SCENE.mjb (MODEL.onnx|hold|random:SEED) EPISODES.txt RESULTS.txt\n";
    return 2;
  } catch (const std::exception& ex) {
    std::cerr << "policy_rollout: " << ex.what() << "\n";
    return 1;
  }
}
