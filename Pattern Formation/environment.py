# based on simple_tag env

import numpy as np
import pygame
from gymnasium.utils import EzPickle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from pettingzoo.mpe._mpe_utils.core import Agent, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

MOUNTAINS = False

CIRCLE_CENTER = (0.5, 0.5)
CIRCLE_RADIUS = 0.2
TRIANGLE_INNER_CENTER = (-0.6, -0.85)
TRIANGLE_OUTER_CENTER = (-0.6, -0.6)
TRIANGLE_INNER_RADIUS = 0.25
TRIANGLE_OUTER_RADIUS = 0.35

outer_triangle_coords = np.array([(TRIANGLE_OUTER_CENTER[0], TRIANGLE_OUTER_CENTER[1] - TRIANGLE_OUTER_RADIUS),
                                 (TRIANGLE_OUTER_CENTER[0] - TRIANGLE_OUTER_RADIUS * np.sqrt(3) / 2, TRIANGLE_OUTER_CENTER[1] + TRIANGLE_OUTER_RADIUS / 2),
                                  (TRIANGLE_OUTER_CENTER[0] + TRIANGLE_OUTER_RADIUS * np.sqrt(3) / 2, TRIANGLE_OUTER_CENTER[1] + TRIANGLE_OUTER_RADIUS / 2)])
inner_triangle_coords = np.array([(TRIANGLE_INNER_CENTER[0], TRIANGLE_INNER_CENTER[1] + TRIANGLE_INNER_RADIUS),
                                  (TRIANGLE_INNER_CENTER[0] + TRIANGLE_INNER_RADIUS * np.sqrt(3) / 2, TRIANGLE_INNER_CENTER[1] - TRIANGLE_INNER_RADIUS / 2),
                                 (TRIANGLE_INNER_CENTER[0] - TRIANGLE_INNER_RADIUS * np.sqrt(3) / 2, TRIANGLE_INNER_CENTER[1] - TRIANGLE_INNER_RADIUS / 2)])


def dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1 - p2) ** 2))


def draw(self):
    # clear screen
    self.screen.fill((255, 255, 255))

    # update bounds to center around agent
    all_poses = [entity.state.p_pos for entity in self.world.entities]
    cam_range = max(np.max(np.abs(np.array(all_poses))), 1)

    # basic env
    pygame.draw.rect(self.screen, (0, 0, 0),
                     (self.width // 2 - self.width // cam_range // 2, self.height // 2 - self.height // cam_range // 2,
                      self.width // cam_range, self.height // cam_range), 1)

    # circle shape
    pygame.draw.circle(self.screen, (200, 200, 200),
                       (CIRCLE_CENTER[0] * self.width // 2 // cam_range + self.width // 2,
                        CIRCLE_CENTER[1] * self.height // 2 // cam_range + self.height // 2),
                       CIRCLE_RADIUS * self.width // 2 // cam_range)

    # triangle shape
    screen_outer_coords = outer_triangle_coords.copy()
    screen_outer_coords[:, 0] = screen_outer_coords[:, 0] * self.width // 2 // cam_range + self.width // 2
    screen_outer_coords[:, 1] = screen_outer_coords[:, 1] * self.height // 2 // cam_range + self.height // 2
    pygame.draw.polygon(self.screen, (200, 200, 200), screen_outer_coords)
    screen_inner_coords = inner_triangle_coords.copy()
    screen_inner_coords[:, 0] = screen_inner_coords[:, 0] * self.width // 2 // cam_range + self.width // 2
    screen_inner_coords[:, 1] = screen_inner_coords[:, 1] * self.height // 2 // cam_range + self.height // 2
    pygame.draw.polygon(self.screen, (255, 255, 255), screen_inner_coords)

    # update geometry and text positions
    text_line = 0
    for e, entity in enumerate(self.world.entities):
        # geometry
        x, y = entity.state.p_pos
        x = (x / cam_range) * self.width // 2 * 0.99
        y = (y / cam_range) * self.height // 2 * 0.99
        x += self.width // 2
        y += self.height // 2
        circle_size = max(entity.size * self.width // 2 // cam_range, 1)
        pygame.draw.circle(
            self.screen, entity.color * 200, (x, y), circle_size
        )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
        pygame.draw.circle(
            self.screen, (0, 0, 0), (x, y), circle_size, 1
        )  # borders

        assert (
                0 < x < self.width and 0 < y < self.height
        ), f"Coordinates {(x, y)} are out of bounds."

        # text
        if isinstance(entity, Agent):
            if entity.silent:
                continue
            if np.all(entity.state.c == 0):
                word = "_"
            elif self.continuous_actions:
                word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                )
            else:
                word = alphabet[np.argmax(entity.state.c)]

            message = entity.name + " sends " + word + "   "
            message_x_pos = self.width * 0.05
            message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
            self.game_font.render_to(
                self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
            )
            text_line += 1


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_agents=10,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_agents=num_agents,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_agents)
        SimpleEnv.draw = draw
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "formation_env"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_agents=10):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            base_name = "agent"
            base_index = i
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.02
            agent.accel = 4.0
            agent.max_speed = 0.05
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            if MOUNTAINS:
                agent.state.p_pos = np.array([np_random.uniform(CIRCLE_CENTER[i] - CIRCLE_RADIUS, CIRCLE_CENTER[i] + CIRCLE_RADIUS) for i in range(world.dim_p)])
            else:
                agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def calc_closest_dist(self, agent, others):
        dists = np.array([dist(agent.state.p_pos, a.state.p_pos) for a in others])
        return dists.min() if len(dists) > 0 else 0

    def point_in_shape(self, point):
        if not MOUNTAINS:
            center_dist = dist(point, CIRCLE_CENTER)
            return center_dist <= CIRCLE_RADIUS
        else:
            point = Point(point)
            outer_triangle = Polygon(outer_triangle_coords)
            inner_triangle = Polygon(inner_triangle_coords)
            return outer_triangle.contains(point) and not inner_triangle.contains(point)

    def reward(self, agent, world):
        if not self.point_in_shape(agent.state.p_pos):
            rew = 0
        else:
            others = world.agents.copy()
            others.remove(agent)
            closest_dist = self.calc_closest_dist(agent, others)
            rew = closest_dist


        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def observation(self, agent, world):
        others = world.agents.copy()
        others.remove(agent)
        closest_ind = np.array([dist(agent.state.p_pos, a.state.p_pos) for a in others]).argmin()
        closest_agent = others[closest_ind]
        return np.concatenate([np.array(TRIANGLE_OUTER_CENTER if MOUNTAINS else CIRCLE_CENTER) - agent.state.p_pos, closest_agent.state.p_pos - agent.state.p_pos])
