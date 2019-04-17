from gym.envs.registration import register

register(id='languageEnv-v0',
         entry_point='gym_language.envs:languageEnv',
)
