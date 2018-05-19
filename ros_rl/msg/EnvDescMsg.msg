# Episodic or continuing?
bool episodic

# If you are going to cut episodes, when should you cut them?
int64 suggestedMaxT

# How many episodes should an agent be given to learn on this env?
int64 suggestedMaxEps

# Reward discount parameter. Only used if episodic (not for continuing tasks)
float64 gamma

# Description of the actions
ros_rl/ThingDescMsg actDesc

# Description of the observations
ros_rl/ThingDescMsg obsDesc

# Various properties one might want to know
float64 deterministicInitialState
float64 minReward	# Set to -INF if not known or not bounded
float64 maxReward	# Set to INF if not known or not bounded
float64 minReturn	# Set to -INF if not known or not bounded
float64 maxReturn	# Set to INF if not known or not bounded

# Recommended plot y-axis
float64 suggestedPlotMinPerformance
float64 suggestedPlotMaxPerformance
