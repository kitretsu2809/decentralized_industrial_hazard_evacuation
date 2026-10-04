"""
Unit Tests for PettingZoo ParallelEnv API and Evacuation Logic.
"""

from pettingzoo.test import parallel_api_test
from envs.evacuation_parallel_env import EvacuationParallelEnv


def test_pettingzoo_parallel_api():
    env = EvacuationParallelEnv(num_pedestrians=15, max_steps=15)
    parallel_api_test(env, num_cycles=10)


def test_evacuation_flow_and_termination():
    env = EvacuationParallelEnv(num_pedestrians=20, max_steps=100)
    obs, infos = env.reset(seed=123)

    assert len(env.agents) > 0
    assert len(env.pedestrians) == 20

    # Step through with valid actions
    for _ in range(50):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

    # Verify tracking counters
    assert (env.total_evacuated + env.total_casualties) <= 20
    assert env.current_step > 0


if __name__ == "__main__":
    test_pettingzoo_parallel_api()
    test_evacuation_flow_and_termination()
    print("All PettingZoo ParallelEnv tests passed!")
