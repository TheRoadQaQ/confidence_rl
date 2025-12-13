# RLCR: Reward Learning with Calibration Regularization

This module implements RLCR (Reward Learning with Calibration Regularization), which directly incorporates calibration into the reward signal during reinforcement learning.

## Mathematical Formulation

### Reward Formula

The core reward formula is:

$$R = R_{acc} - B_{score}$$

where:
- $R_{acc}$ is the accuracy reward: whether the answer is correct (0 or 1)
- $B_{score}$ is the Brier score: $(acc - confidence)^2$

### Brier Score

The Brier score measures calibration error:

$$B_{score} = (acc - confidence)^2$$

where:
- $acc \in \{0, 1\}$ is the actual correctness (1 if answer is correct, 0 otherwise)
- $confidence \in [0, 1]$ is the model's predicted confidence

### Reward at Token Level

At the token level, the reward is applied at the last token of the response:

$$r_T = \begin{cases}
R_{acc} - B_{score} & \text{if format is correct} \\
r_{original} & \text{if format is incorrect}
\end{cases}$$

where:
- $r_T$ is the reward at the last token
- $R_{acc}$ is 1 if the answer is correct, 0 otherwise
- $B_{score} = (acc - confidence)^2$
- Format correctness determines whether to apply the RLCR formula or keep the original reward (usually a format penalty)

## Implementation Details

### Key Components

1. **RayRLCRTrainer**: Extends the base PPO trainer with RLCR reward computation
   - Computes rewards as `acc_reward - brier_score`
   - Applies reward only at the last token of valid responses
   - Handles format errors by keeping original reward
   - Tracks metrics: average Brier score, accuracy reward, confidence, etc.

### Differences from Other Approaches

- **vs. Dynamic Confidence RL**: RLCR does not use Lagrangian multipliers or dynamic constraint optimization. The reward is directly computed as `acc_reward - brier_score`.
- **vs. Split Advantage**: RLCR does not split the advantage computation. It uses a unified reward signal.

### Reward Computation Flow

1. Compute original reward using reward function (rule-based or model-based)
2. Extract accuracy (`acc_tensor`) and confidence (`confidence_tensor`) from reward extra info
3. For each sample with correct format:
   - Calculate `acc_reward = acc` (0 or 1)
   - Calculate `brier_score = (acc - confidence)^2`
   - Set reward at last token: `reward = acc_reward - brier_score`
4. For samples with incorrect format, keep original reward (format penalty)

## Usage

The RLCR reward is automatically applied during training. The configuration is similar to standard PPO training, but the reward computation is modified to incorporate calibration.

### Configuration

The configuration file (`config/rlcr_trainer.yaml`) follows the standard PPO trainer configuration. No special parameters are required for RLCR, as the reward modification is built into the trainer.

### Metrics

The trainer logs the following metrics:
- `rlcr/rewards/mean`: Mean RLCR reward
- `rlcr/avg_brier_score`: Average Brier score (for format-correct samples)
- `rlcr/avg_acc_reward`: Average accuracy reward
- `rlcr/avg_confidence`: Average confidence
- `format/valid_num`: Number of format-correct samples

## Example

For a sample with:
- `acc = 1` (answer is correct)
- `confidence = 0.8`

The reward would be:
- `acc_reward = 1`
- `brier_score = (1 - 0.8)^2 = 0.04`
- `reward = 1 - 0.04 = 0.96`

This encourages the model to:
1. Answer correctly (higher `acc_reward`)
2. Calibrate its confidence well (lower `brier_score`)

