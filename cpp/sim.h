// Simulation core shared by the C++ executables: the same reset, 20 Hz step and success rule as
// teleop_il.env.Env, on a scene compiled by Python and saved as a MuJoCo binary model.
#pragma once
#include <mujoco/mujoco.h>

#include <array>
#include <cmath>
#include <istream>
#include <stdexcept>
#include <string>

namespace sim {

inline const char* kArmJoints[6] = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};

struct Config {
  std::array<double, 6> home{};
  double cube_half = 0, lift_threshold = 0;
  int hold_steps = 0, substeps = 0, episodes = 0;
};

inline std::string expect(std::istream& in, const std::string& key) {
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
  // The policy observation, like Env's: 6 joint positions (5 arm joints and the gripper), then the cube pose.
  void observation(double* out13) const {
    for (int i = 0; i < 6; ++i) out13[i] = d_->qpos[arm_qadr_[i]];
    for (int k = 0; k < 7; ++k) out13[6 + k] = d_->qpos[cube_qadr_ + k];
  }
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

}  // namespace sim
