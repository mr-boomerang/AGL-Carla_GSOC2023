#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Summary of useful helper functions for scenarios
"""

import math
from math import cos, sin

import carla
from agents.navigation.local_planner import RoadOption

import numpy as np
import math


def generate_target_waypoint_list_same_lane(waypoint, distance_same_lane=10, check=True, step_distance=2):
    """
    This methods generates a waypoint list which leads the vehicle to a parallel lane.
    The change input must be 'left' or 'right', depending on which lane you want to change.

    The default step distance between waypoints on the same lane is 2m.
    The default step distance between the lane change is set to 25m.

    @returns a waypoint list from the starting point to the end point on a right or left parallel lane.
    The function might break before reaching the end point, if the asked behavior is impossible.
    """

    plan = []
    plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

    option = RoadOption.LANEFOLLOW

    # Same lane
    distance = 0
    while distance < distance_same_lane:
        next_wps = plan[-1][0].next(step_distance)
        if not next_wps:
            break
        next_wp = next_wps[0]
        distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
        plan.append((next_wp, RoadOption.LANEFOLLOW))

    return plan, None



def get_controls(obs, wp, target_waypts, target_speed = 5.0):
    arr = []
    for waypts in target_waypts:
        dist = waypts[0].transform.location - wp.transform.location
        dist = math.sqrt(dist.x ** 2 + dist.y ** 2)
        arr.append(dist)
    minind = np.argmin(arr)
    velocity = carla.Vector3D(0, 0, 0)
    angular_velocity = carla.Vector3D(0, 0, 0)
    if minind < len(arr) - 1:
        next_location = target_waypts[minind + 1][0].transform.location
        current_speed = math.sqrt(obs.get_velocity().x**2 + obs.get_velocity().y**2)
        direction = next_location - obs.get_location()
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        velocity.x = direction.x / direction_norm * target_speed
        velocity.y = direction.y / direction_norm * target_speed
        print(velocity.x, velocity.y, " target velocity")
        
        # set new angular velocity
        current_yaw = obs.get_transform().rotation.yaw
        delta_yaw = math.degrees(math.atan2(direction.y, direction.x)) - current_yaw

        if math.fabs(delta_yaw) > 360:
            delta_yaw = delta_yaw % 360

        if delta_yaw > 180:
            delta_yaw = delta_yaw - 360
        elif delta_yaw < -180:
            delta_yaw = delta_yaw + 360

        if target_speed == 0:
            angular_velocity.z = 0
        else:
            angular_velocity.z = delta_yaw / (direction_norm / target_speed)

    return velocity, angular_velocity

