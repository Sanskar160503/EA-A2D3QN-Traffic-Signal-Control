# ---- Drop-in replacement for the step() reward section in environment.py ----
#
# Replace your existing reward block inside step() with this logic:
#
#     # Reward: normalized queue penalty + emergency bonus
#     queues      = next_state[:4]
#     total_q     = sum(queues)
#     max_possible = 20 * 4          # tune: max vehicles per lane you expect
#
#     # Normalized penalty in [-1, 0]
#     queue_penalty = -(total_q / max_possible)
#
#     # Emergency bonus: scaled to match normalized range
#     emergency_bonus = 1.0 if emergency_active else 0.0
#
#     reward = queue_penalty + emergency_bonus
#
# This keeps reward in roughly [-1, 1] — no more 40000 vs -5000 swings.
#
# ---- Also add a minimum green time guard inside step() ----
#
# Add this to __init__:
#     self.phase_timer  = 0
#     self.min_green    = 10   # steps before a phase switch is allowed
#
# Replace the RL-controlled signal block with:
#
#     self.phase_timer += 1
#     if action == 1 and self.phase_timer >= self.min_green:
#         current = traci.trafficlight.getPhase(self.tl_id)
#         logic   = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)
#         traci.trafficlight.setPhase(
#             self.tl_id,
#             (current + 1) % len(logic[0].phases)
#         )
#         self.phase_timer = 0   # reset timer after switch
#
# This stops the agent from rapidly toggling signals every step,
# which is a major source of instability.

print("See comments above for drop-in changes to environment.py")
