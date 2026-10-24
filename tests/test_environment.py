"""Regression checks for the notebook environment; no training is executed."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "dqn_gridworld.ipynb"


def load_environment():
    """Execute only Environment and read the map literal from notebook code."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    environment_node = None
    experiment_map = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.ClassDef) and node.name == "Environment":
                environment_node = node
            elif isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "map" for t in node.targets):
                    experiment_map = ast.literal_eval(node.value)
    if environment_node is None or experiment_map is None:
        raise ValueError("Notebook must define Environment and its map literal.")
    namespace = {"np": np, "deepcopy": deepcopy}
    module = ast.Module(body=[environment_node], type_ignores=[])
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace["Environment"], experiment_map


Environment, EXPERIMENT_MAP = load_environment()


def make_map(*placements):
    grid = [["-"] * 6 for _ in range(6)]
    grid[0][0] = "A"
    for row, column, token in placements:
        grid[row][column] = token
    return grid


class EnvironmentTests(unittest.TestCase):
    def test_observation_wraps_both_edges_and_is_one_hot(self):
        env = Environment(make_map(
            (5, 5, "F"), (5, 0, "S"), (5, 1, "R"),
            (0, 5, "C"), (1, 5, "S"), (1, 0, "F"),
        ))
        expected = "FSRCA-SF-"
        self.assertEqual(env.observationStr(), expected)
        observation = env.observation()
        self.assertEqual(observation.shape, (54,))
        cells = observation.reshape(9, 6)
        np.testing.assert_array_equal(cells.sum(axis=1), np.ones(9))
        decoded = "".join(env.values[i] for i in cells.argmax(axis=1))
        self.assertEqual(decoded, expected)

    def test_normal_movement_wraps(self):
        env = Environment(make_map())
        _, reward, done = env.step(0)
        self.assertEqual(env.playerPos, [5, 0])
        self.assertEqual((reward, done), (-1, False))

    def test_normal_swamp_cost_and_departure_clears_terrain(self):
        env = Environment(make_map((0, 1, "S")))
        _, reward, done = env.step(3)
        self.assertEqual((reward, done), (-3, False))
        self.assertEqual(env.map[0][1], "A")
        _, reward, _ = env.step(3)
        self.assertEqual(reward, -1)
        self.assertEqual(env.map[0][1], "-")

    def test_dash_charges_each_swamp_but_only_overwrites_landing(self):
        env = Environment(make_map((0, 1, "S"), (0, 2, "S"), (0, 3, "S")))
        _, reward, done = env.step(7)
        self.assertEqual((reward, done), (-5, False))
        self.assertEqual(env.playerPos, [0, 3])
        self.assertEqual(env.map[0][1:4], ["S", "S", "A"])
        env.step(3)
        self.assertEqual(env.map[0][1:4], ["S", "S", "-"])

    def test_dash_stops_at_intermediate_fire_and_still_adds_beacon(self):
        env = Environment(make_map((0, 1, "S"), (0, 2, "F"), (0, 3, "C")))
        _, reward, done = env.step(7)
        self.assertEqual(env.playerPos, [0, 2])
        self.assertTrue(done)
        self.assertEqual(reward, -999)  # -2 dash -1 swamp -1000 fire +4 beacon.
        self.assertEqual(env.totalCrystals, 1)
        self.assertEqual(env.map[0][1:4], ["S", "A", "C"])

    def test_dash_crosses_crystal_but_collects_only_landing_crystal(self):
        env = Environment(make_map((0, 1, "C"), (0, 3, "R")))
        _, reward, done = env.step(7)
        self.assertEqual((reward, done), (5, False))  # -2 +5 pickup +2 beacon.
        self.assertEqual(env.totalCrystals, 1)
        self.assertEqual(env.map[0][1], "C")
        self.assertEqual(env.map[0][3], "A")

    def test_final_crystal_bonus_has_no_remaining_beacon(self):
        for action, column, expected_reward in [(3, 1, 1004), (7, 3, 1003)]:
            with self.subTest(action=action):
                env = Environment(make_map((0, column, "C")))
                _, reward, done = env.step(action)
                self.assertEqual((reward, done), (expected_reward, True))
                self.assertEqual(env.totalCrystals, 0)
                self.assertEqual(env.beacon_reward(), 0)

    def test_beacon_uses_toroidal_manhattan_distance(self):
        for row, column, reward in [(5, 0, 4), (5, 5, 2), (4, 5, 0)]:
            with self.subTest(crystal=(row, column)):
                env = Environment(make_map((row, column, "C")))
                self.assertEqual(env.beacon_reward(), reward)

    def test_beacon_retargets_after_crystal_removal(self):
        env = Environment(make_map((0, 1, "C"), (0, 3, "R")))
        self.assertEqual(env.beacon_reward(), 4)
        _, reward, done = env.step(3)
        self.assertEqual((reward, done), (6, False))
        self.assertEqual(env.beacon_reward(), 2)
        self.assertEqual(env.totalCrystals, 1)

    def test_reset_restores_terrain_crystals_and_observation(self):
        grid = make_map((0, 1, "S"), (0, 3, "C"), (1, 0, "R"))
        env = Environment(grid)
        initial_observation = env.observation().copy()
        env.step(7)
        self.assertEqual(env.totalCrystals, 1)
        reset_observation = env.reset()
        self.assertEqual(env.map, grid)
        self.assertEqual(env.totalCrystals, 2)
        self.assertEqual(env.playerPos, [0, 0])
        np.testing.assert_array_equal(reset_observation, initial_observation)

    def test_saved_51_step_trace_replays_to_minus_33(self):
        # Actions/rewards transcribed from the supplied notebook's saved rollout.
        # This checks environment consistency, not reproduction of trained weights.
        actions = [2, 5, 2, 5] + [3, 5, 2, 5] * 11 + [3, 5, 2]
        expected_rewards = [-1, 3, 1, -3] + [-1, -2, 3, -3] * 11 + [-1, -2, 3]
        env = Environment(deepcopy(EXPERIMENT_MAP))
        rewards = []
        for step, action in enumerate(actions, start=1):
            _, reward, done = env.step(action)
            self.assertFalse(done, f"Unexpected terminal state at step {step}")
            rewards.append(reward)
        self.assertEqual(rewards, expected_rewards)
        self.assertEqual(len(rewards), 51)
        self.assertEqual(sum(rewards), -33)
        self.assertEqual(env.totalCrystals, 1)
        self.assertEqual(env.playerPos, [0, 0])


if __name__ == "__main__":
    unittest.main()
