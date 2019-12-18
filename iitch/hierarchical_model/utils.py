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
"""Util functions for generating data to be input into the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime


import numpy as np
import pandas as pd
import scipy
from google.cloud import bigquery

ObservedData = collections.namedtuple('ObservedData', [
    'trap_indices', 'trap_durations', 'm_counts', 'f_counts', 'baited_traps',
    'trap_lat', 'trap_lng', 'trap_collection_day', 'release_tensor',
    'obs_locations', 'num_locations', 'time_steps', 'squared_dist_matrix',
    'lat_seq', 'lng_seq', 'time_seq'
])

Partitioning = collections.namedtuple('Partitioning',
                                      ['time_seq', 'lng_seq', 'lat_seq'])


def get_release_events(
    site_name,
    min_time,
    max_time,
    project='google.com:debugger',
    table='plx.google:debug_analytics_jobs.fat_mosquito_counts_with_gps.all'):
  """Get release events from within site polygon during a time window.

  Args:
      site_name: String of a name from debug_analytics_jobs.fat_planner_site,
        e.g. "Loma Vista". Release events returned will be from within this
        site's polygon.
      min_time: String of a timestamp that can be parsed by GoogleSQL, e.g.
        "2018-10-17 00:00:00 America/Los_Angeles". Release events returned will
          be from after this time.
      max_time: String of a timestamp that can be parsed by GoogleSQL, e.g.
        "2018-10-17 15:00:00 America/Los_Angeles". Release events returned will
          be from before this time.  Queries
          debug_analytics_jobs.fat_planner_site and
          debug_analytics_jobs.ssf_mosquito_counts_with_gps for events.
      project: BigQuery project that contains the data.
      table: Data table to query.

  Returns:
      Pandas dataframe with columns ~ <DrivethroughId, Latitude, Longitude,
      MosquitoExitTime, MosquitoesReleased>
  """
  client = bigquery.Client(project=project)
  job_config = bigquery.QueryJobConfig()
  job_config.use_legacy_sql = False
  job_config.use_query_cache = True
  job_config.flatten_results = True

  query = """
  with site_polygon as (
  select
    Geography as polygon
  from
    debug_analytics_jobs.fat_planner_site
  where
    name = "{site_name}"
  )
  select
    DrivethroughId,
    Lat lat,
    Lng lng,
    MosquitoExitTime,
    MosquitoesReleased
  from
    `{table}`
  cross join
    site_polygon
  where
    st_contains(site_polygon.polygon, st_geogpoint(lng, lat)) and
    MosquitoExitTime >= timestamp "{min_time}" and
    MosquitoExitTime <= timestamp "{max_time}"
  order by
    MosquitoExitTime asc
  """.format(
      site_name=site_name, min_time=min_time, max_time=max_time, table=table)
  df = client.query(query, job_config=job_config).to_dataframe()
  df['time'] = [
      datetime.datetime.fromtimestamp(_ / 1000000) for _ in df.MosquitoExitTime
  ]
  return df


def get_trap_data(site_name,
                  min_time,
                  max_time,
                  project='google.com:debugger',
                  table='plx.google:debug_analytics_jobs.adult_sortings.all'):
  """Get trap data from within site polygon during a time window.

  Args:
      site_name: String of a name from debug_analytics_jobs.fat_planner_site,
        e.g. "Loma Vista". Trap results returned will be from within this site's
        polygon.
      min_time: String of a timestamp that can be parsed by GoogleSQL, e.g.
        "2018-10-17 00:00:00 America/Los_Angeles". Trap results returned will be
          from after this time.
      max_time: String of a timestamp that can be parsed by GoogleSQL, e.g.
        "2018-10-17 15:00:00 America/Los_Angeles". Trap results returned will be
          from before this time.  Queries debug_analytics_jobs.fat_planner_site
          and debug_analytics_jobs.adult_sortings for trap results.
      project: BigQuery project that contains the data.
      table: Data table to query.

  Returns:
      Pandas dataframe with columns ~ <MaleAedesCount, FemaleAedesCount,
      CollectionDay, PreviousCollectionDay, TrapLat, TrapLng, UsedCO2, TrapId>
  """
  client = bigquery.Client(project=project)
  job_config = bigquery.QueryJobConfig()
  job_config.use_legacy_sql = False
  job_config.use_query_cache = True
  job_config.flatten_results = True

  query = """
  with site_polygon as (
  select
    Geography as polygon
  from
    debug_analytics_jobs.fat_planner_site
  where
    name = "{site_name}"
  )
  select
    MaleAedesCount,
    FemaleAedesCount,
    CollectionDay,
    PreviousCollectionDay,
    TrapLat,
    TrapLng,
    UsedCO2,
    TrapId
  from
    `{table}`
  cross join
    site_polygon
  where
    st_contains(site_polygon.polygon, st_geogpoint(TrapLng, TrapLat)) and
    PreviousCollectionDay >= timestamp "{min_time}" and
    PreviousCollectionDay <= timestamp "{max_time}"
  order by
    CollectionDay asc
  """.format(
      site_name=site_name, min_time=min_time, max_time=max_time, table=table)
  df = client.query(query, job_config=job_config).to_dataframe()
  df['CollectionDay'] = [
      datetime.datetime.fromtimestamp(_ / 1000000) for _ in df.CollectionDay
  ]
  df['PreviousCollectionDay'] = [
      datetime.datetime.fromtimestamp(_ / 1000000)
      for _ in df.PreviousCollectionDay
  ]
  return df


def generate_observation_data(trap_data,
                              release_events,
                              partitioning,
                              id_col='TrapId',
                              lat_col='TrapLat',
                              lng_col='TrapLng',
                              collection_day_col='CollectionDay',
                              prev_collection_day_col='PreviousCollectionDay',
                              used_co2_col='UsedCO2',
                              male_col='MaleAedesCount',
                              female_col='FemaleAedesCount',
                              release_lat_col='lat',
                              release_lng_col='lng',
                              release_time_col='time',
                              num_mosquitoes_col='MosquitoesReleased',
                              dtype=np.float32):
  """Convert trap data into format for model.

  Arguments:
    trap_data: trap data generated by get_trap_data.
    release_events: release event data generated by get_release_events.
    partitioning: Partitioning that contains pandas range intervals that define
      how the model is discretized.
    id_col: String indicating the trap data column for trap id.
    lat_col: String indicating the trap data column for trap lat.
    lng_col: String indicating the trap data column for trap lng.
    collection_day_col: String indicating the trap data column for trap
      collection day.
    prev_collection_day_col: String indicating the trap data column for previous
      trap collection day.
    used_co2_col: String indicating the trap data column for if CO2 bait was
      used.
    male_col: String indicating the trap data column for number of males caught.
    female_col: String indicating the trap data column for number of females
      caught.
    release_lat_col: String indicating the release_events column for release
      latitude.
    release_lng_col: String indicating the release_events column for release
      longitude.
    release_time_col: String indicating the release_events column for release
      time.
    num_mosquitoes_col: String indicating the release_events column for number
      of mosquitoes released.
    dtype: dtype used for floating point data.

  Returns:
    ObservedData for input to the model
  """

  max_y_idx = len(partitioning.lat_seq) - 1
  sqrt_num_locations = len(partitioning.lat_seq)

  d = trap_data.copy().sort_values(
      [id_col, lat_col, lng_col, collection_day_col])

  x = np.array(d[lng_col].apply(partitioning.lng_seq.get_loc), dtype=np.int32)
  y = np.array(d[lat_col].apply(partitioning.lat_seq.get_loc), dtype=np.int32)
  t = np.array(
      d[collection_day_col].apply(partitioning.time_seq.get_loc),
      dtype=np.int32)
  idx = (max_y_idx - y) * sqrt_num_locations + x
  m_counts = np.array(d[male_col])
  f_counts = np.array(d[female_col])
  baited_traps = np.array(d[used_co2_col], dtype=np.int32)
  trap_durations = np.array(
      (d[collection_day_col] - d[prev_collection_day_col]).dt.days)
  trap_collection_day = np.array(d[collection_day_col])
  trap_lat = np.array(d[lat_col])
  trap_lng = np.array(d[lng_col])

  trap_indices = []
  obs_locations = []

  # map observations into index so that we can index into an "abundance vector"
  # for each observation during model fitting.
  obs_location_to_id = {}
  i = 0
  t_idx = zip(t, idx)
  for (curr_t, curr_idx) in t_idx:
    if (curr_t, curr_idx) not in obs_location_to_id:
      obs_location_to_id[(curr_t, curr_idx)] = i
      obs_locations.append((curr_t, curr_idx))
      i += 1
    trap_indices.append(obs_location_to_id[(curr_t, curr_idx)])

  squared_dist_matrix = generate_squared_dist_matrix(
      sqrt_num_locations, dtype=dtype)

  release_tensor = generate_release_tensor(
      release_events,
      partitioning,
      lat_col=release_lat_col,
      lng_col=release_lng_col,
      time_col=release_time_col,
      num_mosquitoes_col=num_mosquitoes_col,
      dtype=dtype)

  obs_dat = ObservedData(
      trap_indices=trap_indices,
      trap_durations=trap_durations,
      m_counts=m_counts,
      f_counts=f_counts,
      baited_traps=baited_traps,
      trap_lat=trap_lat,
      trap_lng=trap_lng,
      trap_collection_day=trap_collection_day,
      release_tensor=release_tensor,
      obs_locations=obs_locations,
      num_locations=sqrt_num_locations**2,
      time_steps=len(partitioning.time_seq),
      squared_dist_matrix=squared_dist_matrix,
      lat_seq=partitioning.lat_seq,
      lng_seq=partitioning.lng_seq,
      time_seq=partitioning.time_seq)

  return obs_dat


def generate_squared_dist_matrix(sqrt_num_locations, dtype=np.float32):
  """Generate a squared distance matrix of size sqrt_num_locations**2."""
  num_locations = sqrt_num_locations**2
  loc_coords = np.array([(x, y)  # pylint: disable=g-complex-comprehension
                         for x in np.arange(sqrt_num_locations)
                         for y in np.arange(sqrt_num_locations)],
                        dtype=dtype)
  squared_dist_matrix = np.array(
      scipy.spatial.distance_matrix(loc_coords, loc_coords),
      dtype=dtype).reshape((num_locations, num_locations))**2
  return squared_dist_matrix


def generate_release_tensor(release_events,
                            partitioning,
                            lat_col='lat',
                            lng_col='lng',
                            time_col='time',
                            num_mosquitoes_col='MosquitoesReleased',
                            dtype=np.float32):
  """Generate the TxN tensor that defines when and where releases occurred."""
  d = release_events.copy()
  max_y_idx = len(partitioning.lat_seq) - 1
  release_tensor = np.zeros(
      [len(partitioning.time_seq),
       len(partitioning.lng_seq)**2, 1],
      dtype=dtype)
  r = pd.DataFrame()
  r['x'] = d[lng_col].apply(partitioning.lng_seq.get_loc)
  r['y'] = d[lat_col].apply(partitioning.lat_seq.get_loc)
  r['t'] = d[time_col].apply(partitioning.time_seq.get_loc)
  r['idx'] = (max_y_idx - r['y']) * len(partitioning.lng_seq) + r['x']
  r['num_mosquitoes'] = d[num_mosquitoes_col]

  for e in r.itertuples(index=False):
    release_tensor[e.t, e.idx] += e.num_mosquitoes
  return release_tensor


def calculate_data_partitioning(obs_data,
                                release_events,
                                space_res,
                                time_res,
                                obs_collection_day_col='CollectionDay',
                                obs_lat_col='TrapLat',
                                obs_lng_col='TrapLng',
                                release_time_col='time',
                                release_lat_col='lat',
                                release_lng_col='lng'):
  """Calculate data partitioning sequences for model.

  Calculate the sequences of:
    - The grid of observations and releases
    - Time steps
   as defined by the given spatial and temporal resolutions

  Arguments:
    obs_data: DataFrame generated by get_trap_data that contains the observed
      trapping events
    release_events: DataFrame generated by get_release_events that contains all
      mosquito release events
     space_res: Integer defining the edge length of the square partitioning
       grid.
     time_res: Integer defining the number of time steps in the temporal
       partitioning sequence.
     obs_collection_day_col: String indicating the obs_data column for
       collection day.
     obs_lat_col: String indicating the obs_data column for trap latitude.
     obs_lng_col: String indicating the obs_data column for trap longitude.
     release_time_col: String indicating the release_events column for release
       time.
     release_lat_col: String indicating the release_events column for release
       latitude.
     release_lng_col: String indicating the release_events column for release
       longitude.

  Returns:
    Partitioning containing Pandas range intervals that define the sequence of
    latitude, longitude, and time steps to discretize space and time.

  """
  min_release_time = release_events[release_time_col].min()
  max_release_time = release_events[release_time_col].max()
  min_collection_day = obs_data[obs_collection_day_col].min()
  max_collection_day = obs_data[obs_collection_day_col].max()
  min_time = min(min_release_time, min_collection_day) - datetime.timedelta(1)
  max_time = max(max_release_time, max_collection_day) + datetime.timedelta(1)
  time_seq = pd.interval_range(start=min_time, end=max_time, periods=time_res)

  min_release_lat = release_events[release_lat_col].min()
  max_release_lat = release_events[release_lat_col].max()
  min_collection_lat = obs_data[obs_lat_col].min()
  max_collection_lat = obs_data[obs_lat_col].max()
  min_lat = min(min_release_lat, min_collection_lat)
  max_lat = max(max_release_lat, max_collection_lat)
  min_lat = min_lat - .001 * np.abs(min_lat - max_lat)
  max_lat = max_lat + .001 * np.abs(max_lat - max_lat)
  lat_seq = pd.interval_range(min_lat, max_lat, space_res)

  min_release_lng = release_events[release_lng_col].min()
  max_release_lng = release_events[release_lng_col].max()
  min_collection_lng = obs_data[obs_lng_col].min()
  max_collection_lng = obs_data[obs_lng_col].max()
  min_lng = min(min_release_lng, min_collection_lng)
  max_lng = max(max_release_lng, max_collection_lng)
  min_lng = min_lng - .001 * np.abs(min_lng - max_lng)
  max_lng = max_lng + .001 * np.abs(min_lng - max_lng)
  lng_seq = pd.interval_range(min_lng, max_lng, space_res)

  return Partitioning(time_seq, lng_seq, lat_seq)
