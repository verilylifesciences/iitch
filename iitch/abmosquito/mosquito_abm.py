# Copyright 2019 Verily Life Sciences LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of an agent based model for mosquito movement and trapping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from six.moves import range


class EnvironmentModel(object):
  """Context container for execution of the model."""

  def __init__(self, script, dt=.01):
    """Initialize the model with a script of actions.

       Actions are of the following form:
          {
            'time': int,
            'action': model_action,
            'parameters': model_action_parameters
          }
        'time' is the integer timestep the action will occur
        'action' is an action supported by the environment
        'parameters' is a dictionary passed to the action

    Arguments:
      script: List of actions that will be taken by the model during
        initialization.
      dt: The timestep discretization size of the model.
    """

    self.traps = TrapAgentsContainer(environment=self)
    self.mosquitoes = MosquitoAgentsContainer(environment=self)
    self.script = self.script = sorted(script, key=lambda x: x['time'])
    self.dt = dt
    self.current_time = 0
    self.mosquito_count = 0
    self.trap_count = 0

  def add_traps(self, traps):
    """Add traps to the environment.

    Arguments:
      traps: TrapAgentsContainer
    """
    self.traps.add_traps(**traps)

  def remove_trap(self, trap_uid):
    """Remove traps from the environment.

    Arguments:
      trap_uid: unique identifier of the trap
    """
    self.traps.remove_trap(trap_uid)

  def add_mosquitoes(self, mosquitoes):
    """Add mosquitoes to the environment.

    Arguments:
      mosquitoes: MosquitoAgentsContainer
    """
    self.mosquitoes.add_mosquitoes(**mosquitoes)

  def advance_time(self):
    """Advance the model through the initialization script and then one step."""
    while self.script and self.current_time >= self.script[0]['time']:
      command = self.script.pop(0)
      action = getattr(self, command['action'])
      action(command['parameters'])

    self.current_time += self.dt

    if self.mosquito_count:
      self.mosquitoes.pre_step()
    if self.trap_count:
      self.traps.pre_step()

    if self.mosquito_count:
      self.mosquitoes.step(self.dt)
    if self.trap_count:
      self.traps.step(self.dt)

    if self.mosquito_count:
      self.mosquitoes.post_step()
    if self.trap_count:
      self.traps.post_step()


class AgentsContainer(object):
  """Base class implementing an Agent Container."""

  def __init__(self, environment):
    self.environment = environment
    self.x_coords = None
    self.y_coords = None
    self.start_times = None
    self.prev_x_coords = None
    self.prev_y_coords = None

  def add_agents(self, x_coords, y_coords):
    """Add agents to the environment.

    Arguments:
      x_coords: x coordinates of where to add the agents
      y_coords: y coordinates of where to add the agents
    """
    if self.x_coords is not None:
      self.x_coords = np.concatenate((self.x_coords, x_coords))
    else:
      self.x_coords = np.array(x_coords)

    if self.y_coords is not None:
      self.y_coords = np.concatenate((self.y_coords, y_coords))
    else:
      self.y_coords = np.array(y_coords)

    if self.start_times is not None:
      self.start_times = np.concatenate(
          (self.start_times,
           np.ones(x_coords.shape) * self.environment.current_time))
    else:
      self.start_times = (
          np.ones(self.x_coords.shape) * self.environment.current_time)

    self.prev_x_coords = np.copy(self.x_coords)
    self.prev_y_coords = np.copy(self.y_coords)

  def pre_step(self):
    """Function called before each step."""
    pass

  def step(self, dt):
    """Function called during each step.

    Arguments:
      dt: The time discretization over which the function is applied.
    """
    pass

  def post_step(self):
    """Function called after each step."""
    pass


class TrapAgentsContainer(AgentsContainer):
  """Trap Agents Container controls a collection of traps.

  Traps are composed of an attractiveness factor, a capture radius, and a unique
  identifier
  """

  def __init__(self, environment):
    """Initialize a TrapAgentsContainer.

    Arguments:
       environment: The environment in which the TrapAgentsContainer exists
    """
    super(TrapAgentsContainer, self).__init__(environment)
    self.attractiveness_vec = None
    self.capture_radius_vec = None
    self.trap_ids = None

  def add_traps(self, x_coords, y_coords, trap_ids, attractiveness_vec,
                capture_radius_vec):
    """Add traps to the trap agents container.

    Arguments:
      x_coords: The x coordinates of the traps
      y_coords: The y coordinates of the traps
      trap_ids: The unique IDs of the traps
      attractiveness_vec: Attractiveness constant for each trap
      capture_radius_vec: Radius from the trap point for capture
    """
    self.add_agents(x_coords, y_coords)

    if self.attractiveness_vec is not None:
      self.attractiveness_vec = np.concatenate(
          (self.attractiveness_vec, attractiveness_vec))
    else:
      self.attractiveness_vec = attractiveness_vec

    if self.capture_radius_vec is not None:
      self.capture_radius_vec = np.concatenate(
          (self.capture_radius_vec, capture_radius_vec))
    else:
      self.capture_radius_vec = capture_radius_vec

    if self.trap_ids is not None:
      self.trap_ids = np.concatenate((self.trap_ids, trap_ids))
    else:
      self.trap_ids = trap_ids

    self.environment.trap_count += x_coords.shape[0]

  def remove_trap(self, trap_uid):
    """Remove a trap from the trap agents container.

    Arguments:
      trap_uid: The unique identifier of the trap when initialized
    """
    trap_idx = np.where(self.trap_ids == trap_uid)
    self.x_coords = np.delete(self.x_coords, trap_idx, 0)
    self.y_coords = np.delete(self.y_coords, trap_idx, 0)
    self.attractiveness_vec = np.delete(self.attractiveness_vec, trap_idx, 0)
    self.capture_radius_vec = np.delete(self.capture_radius_vec, trap_idx, 0)
    self.trap_ids = np.delete(self.trap_ids, trap_idx, 0)
    self.environment.trap_count -= 1

  def pre_step(self):
    """Currently unimplemented."""
    pass

  def step(self, dt):
    """Apply a single step of the model.

    Arguments:
      dt: The time discretization over which the model is run.
    """
    self.attract(dt)

  def post_step(self):
    """Trap mosquitoes that are within range after each time step."""
    self.trap()

  def trap(self):
    """Trap mosquitoes that are within range of the trap."""
    if self.environment.mosquito_count:
      n_traps = self.x_coords.shape[0]
      for i in range(n_traps):
        center_x = self.x_coords[i]
        center_y = self.y_coords[i]
        trap_id = self.trap_ids[i]
        capture_distance = self.capture_radius_vec[i]
        self.environment.mosquitoes.trap(center_x, center_y, capture_distance,
                                         trap_id)

  def attract(self, dt):
    """Attract mosquitoes to the trap over time dt.

    Arguments:
      dt: The time discretization over which the attractiveness is applied
    """
    if self.environment.mosquito_count:
      n_traps = self.x_coords.shape[0]
      for i in range(n_traps):
        center_x = self.x_coords[i]
        center_y = self.y_coords[i]
        attractiveness = self.attractiveness_vec[i]

        x = self.environment.mosquitoes.prev_x_coords - center_x
        y = self.environment.mosquitoes.prev_y_coords - center_y
        dist = np.sqrt(x**2 + y**2)
        force = -dt * attractiveness / (dist)

        delta_x = x / dist * force
        delta_y = y / dist * force
        self.environment.mosquitoes.move(delta_x, delta_y)


class MosquitoAgentsContainer(AgentsContainer):
  """Mosquito Agents Container controls a collection of mosquitoes.

  Mosquitoes are composed of an average movement distance and a lifespan.
  """

  def __init__(self, environment):
    """Initialize the mosquito agents container in the environment.

    Arguments:
      environment: The environment in which the MosquitoAgentsContainer exists
    """
    super(MosquitoAgentsContainer, self).__init__(environment)
    self.move_sigma_vec = None
    self.lifespan_vec = None
    self.alive_mosquitoes = None
    self.dead_mosquitoes = None
    self.trapped_mosquitoes = None
    self.trap_ids = None

  def add_mosquitoes(self, x_coords, y_coords, move_sigma_vec, lifespan_vec):
    """Add mosquitoes to the environment.

    Arguments:
      x_coords: the initial x coordinate position of mosquitoes
      y_coords: the initial y coordinate position of mosquitoes
      move_sigma_vec: the average distance traveled by the mosquito per dt
      lifespan_vec: the lifespan of the mosquito
    """
    self.add_agents(x_coords, y_coords)
    if self.move_sigma_vec is not None:
      self.move_sigma_vec = np.concatenate(
          (self.move_sigma_vec, move_sigma_vec))
    else:
      self.move_sigma_vec = move_sigma_vec

    if self.lifespan_vec is not None:
      self.lifespan_vec = np.concatenate((self.lifespan_vec, lifespan_vec))
    else:
      self.lifespan_vec = lifespan_vec

    if self.alive_mosquitoes is not None:
      self.alive_mosquitoes = np.concatenate(
          (self.alive_mosquitoes, np.ones(x_coords.shape)))
    else:
      self.alive_mosquitoes = np.ones(x_coords.shape)

    if self.dead_mosquitoes is not None:
      self.dead_mosquitoes = np.concatenate(
          (self.dead_mosquitoes, np.zeros(x_coords.shape)))
    else:
      self.dead_mosquitoes = np.zeros(x_coords.shape)

    if self.trapped_mosquitoes is not None:
      self.trapped_mosquitoes = np.concatenate(
          (self.trapped_mosquitoes, np.zeros(x_coords.shape)))
    else:
      self.trapped_mosquitoes = np.zeros(x_coords.shape)

    if self.trap_ids is not None:
      self.trap_ids = np.concatenate(
          (self.trap_ids, np.ones(x_coords.shape) * -1))
    else:
      self.trap_ids = np.ones(x_coords.shape) * -1

    self.environment.mosquito_count += x_coords.shape[0]

  def pre_step(self):
    """Record the mosquitoes previous location."""
    self.prev_x_coords = np.copy(self.x_coords)
    self.prev_y_coords = np.copy(self.y_coords)

  def step(self, dt):
    """Step the model forward.

    Move mosquitoes randomly in a brownian motion fashion.

    Arguments:
      dt: The time discretization over which movement occurs.
    """
    self.random_move(dt)

  def post_step(self):
    """After each step, mosquitoes die if they have exceeded their lifespan."""
    self.die()

  def random_move(self, dt):
    """Move randomly in a brownian motion fashion.

    Arguments:
      dt: The time discretization over which the movement occurs.
    """
    n_mosqs = self.move_sigma_vec.shape[0]
    delta_x = self.move_sigma_vec * np.random.normal(0, 1,
                                                     n_mosqs) * np.sqrt(dt)
    delta_y = self.move_sigma_vec * np.random.normal(0, 1,
                                                     n_mosqs) * np.sqrt(dt)
    self.move(delta_x, delta_y)

  def die(self):
    """Mosquitoes that have exceeded their lifespan are marked as dead."""
    self.dead_mosquitoes = np.logical_and(
        np.less(self.lifespan_vec, self.environment.current_time),
        np.logical_not(self.trapped_mosquitoes))
    self.alive_mosquitoes = np.logical_and(
        np.logical_not(self.dead_mosquitoes),
        np.logical_not(self.trapped_mosquitoes))

  def trap(self, center_x, center_y, capture_distance, trap_uid):
    """Mosquitoes that are within the range of a trap are trapped.

    Arguments:
      center_x: The x coordinate of traps.
      center_y: The y coordinate of traps.
      capture_distance: The distance from the trap that results in a capture.
      trap_uid: The unique ids of the traps.
    """
    dist = np.sqrt((self.x_coords - center_x)**2 +
                   (self.y_coords - center_y)**2)
    in_trap = np.less(dist, capture_distance)
    self.trapped_mosquitoes = np.logical_or(in_trap, self.trapped_mosquitoes)
    self.trap_ids = (self.trap_ids * np.logical_not(in_trap)) + (
        in_trap * trap_uid)
    self.alive_mosquitoes = np.logical_and(
        np.logical_not(self.dead_mosquitoes),
        np.logical_not(self.trapped_mosquitoes))

  def move(self, delta_x, delta_y):
    """Apply a movement vector to each mosquito.

    Arguments:
      delta_x: The change in position in the x direction.
      delta_y: The change in position in the y direction.
    """
    self.x_coords += delta_x * self.alive_mosquitoes
    self.y_coords += delta_y * self.alive_mosquitoes
