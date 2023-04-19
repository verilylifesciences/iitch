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

# Agent based modeling of mosquito population and animations

load("//tools/build_defs/build_test:build_test.bzl", "build_test")

package(default_applicable_licenses = ["//third_party/py/iitch:license"])

licenses(["notice"])

py_library(
    name = "abmosquito",
    srcs = [
        "animation.py",
        "mosquito_abm.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("scipy"),
        requirement("six"),
    ],
)

build_test(
    name = "abmosquito_build_test",
    targets = [":abmosquito"],
)
