# Lint as: python2, python3
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
"""Class for animating an agent based model using matplotlib animation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from six.moves import range


class AnimatedMosquitoModel(object):
  """Class for animating a mosquito agent based model run.
  """

  def __init__(self, environment, steps_per_frame=3, frames=200, interval=50,
               axis=None):
    """Generates the matplotlib animation from a given environment.

    Arguments:
      environment: initialized EnvironmentModel
      steps_per_frame: Number of time steps in the model per frame of animation
      frames: Total number of frames to generate
      interval: Number of milliseconds between frames in the animation
      axis: Domain and range of plot: [lower_x, upper_x, lower_y, upper_y]
    """
    if not axis:
      self.axis = [-150, 150, -150, 150]
    else:
      self.axis = axis
    self.environment = environment
    self.steps_per_frame = steps_per_frame
    self.fig, self.ax = plt.subplots()
    self.stream = self._data_stream()
    self.ani = animation.FuncAnimation(self.fig,
                                       self._update,
                                       interval=interval,
                                       init_func=self._setup_plot,
                                       blit=False,
                                       frames=frames)

  def _setup_plot(self):
    """Internal function to setup the plot.

    Returns:
      The initialized plot
    """
    x, y, c = next(self.stream).T
    # init color to full range so that animations work
    c = np.zeros(x.shape)
    c[0] = 1
    self.scat = self.ax.scatter(x, y, c=c, cmap="jet")
    self.ax.axis(self.axis)
    self.ax.set_aspect(1)
    return (self.scat,)

  def _data_stream(self):
    """Generator that yields data to the animation.

    Yields:
      yields data for subsequent frames
    """
    while True:
      for _ in range(self.steps_per_frame):
        self.environment.advance_time()
      x_coords = self.environment.mosquitoes.x_coords
      y_coords = self.environment.mosquitoes.y_coords
      # map state [alive, trapped, dead] to colors [.1, .5, .9]
      c = (self.environment.mosquitoes.alive_mosquitoes * .1 +
           self.environment.mosquitoes.trapped_mosquitoes * .5 +
           self.environment.mosquitoes.dead_mosquitoes * .9
          )
      yield np.c_[x_coords, y_coords, c]

  def _update(self, _):
    """Internal function for updating the plot.

    Returns:
      The updated plot
    """
    data = next(self.stream)
    self.scat.set_offsets(data[:, :2])
    self.scat.set_array(data[:, 2])
    return (self.scat,)
