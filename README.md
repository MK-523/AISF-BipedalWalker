# AISF-BipedalWalker
progress with AISF bipedal walker project

wrappers
- ResearchRewardWrapper: solution to Sparse Reward Trap, adds posture penalty and velocity bonus
- make_research_env: handles vectorization and Observation Normalization

expA_entropy
- tests if increasing entropy coefficient to 0.05 breaks the fear of falling and forces exploration

expB_shaping
- evaluates specific impact of custom reward formula on the agent's stability

expC_lr
- tests how a lower learning rate stabilizes the agent's legs

final_agent
- deep architecture that handles complex lidar to joint mapping
- this combines hyperparameters with the best learning rate and entropy settings from above

evaluate
- visual verification with imageio to record the walker

requirements
- outlines required libraries
