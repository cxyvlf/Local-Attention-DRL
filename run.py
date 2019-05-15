import numpy as np
from schema_games.breakout import games
from schema_games.printing import blue

env_args = {
    'return_state_as_image': True,
}

env = games.StandardBreakout(**env_args)
n_actions = env.action_space.n

print env.observation_space.shape
print env.action_space
print n_actions

env.reset()

for i_episode in range(3000):
    print "Episode:", i_episode,"\n"
    env.reset()
    while True:
        env.render()
        action = np.random.randint(0, n_actions)
        observation, reward, done, info = env.step(action)
        

        if done:
            break

