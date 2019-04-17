# gym-NAME/
#     README.md
#     setup.py
#     gym_NAME/
#         __init__.py
#         envs/
#             __init__.py
#             NAME_env.py

NAME=$1
VERSION=$2

mkdir -p gym-$NAME/gym_$NAME/envs/

printf "from setuptools import setup\n\n" > gym-$NAME/setup.py
echo "setup(name='gym_${NAME}'," >>  gym-$NAME/setup.py
echo "      version='${VERSION}'," >> gym-$NAME/setup.py
echo "      install_requires=['gym', 'numpy', 'pygame']" >> gym-$NAME/setup.py
printf ")\n" >> gym-$NAME/setup.py

printf "from gym.envs.registration import register\n\n" > gym-$NAME/gym_$NAME/__init__.py
echo "register(id='${NAME}Env-v0'," >>  gym-$NAME/gym_$NAME/__init__.py
echo "         entry_point='gym_${NAME}.envs:${NAME}Env'," >>  gym-$NAME/gym_$NAME/__init__.py
printf ")\n" >>  gym-$NAME/gym_$NAME/__init__.py

echo "from gym_${NAME}.envs.${NAME}_env import ${NAME}Env" > gym-$NAME/gym_$NAME/envs/__init__.py

echo "import gym" > gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "from gym import error, spaces, utils"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "from gym.utils import seeding"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo   >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "class ${NAME}Env(gym.Env):"    >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "    metadata = {'render.modes': ['human']}"     >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "    def __init__(self):"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "        pass"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo   >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "    def step(self, action):"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "        pass"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo   >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "    def reset(self):"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "        pass"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo   >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "    def render(self, mode='human', close=False):"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py
echo "        pass"  >> gym-$NAME/gym_$NAME/envs/${NAME}_env.py