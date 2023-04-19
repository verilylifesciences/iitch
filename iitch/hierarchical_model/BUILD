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

# Implementation of a hierarchical population dispersion model.

load("//tools/build_defs/build_test:build_test.bzl", "build_test")

package(default_applicable_licenses = ["//third_party/py/iitch:license"])

licenses(["notice"])  # Apache License 2.0

py_library(
    name = "hierarchical_model",
    srcs = [
        "tfp_model_v1.py",
        "tfp_model_v2.py",
        "utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/google/cloud/bigquery",
        requirement("numpy"),
        requirement("pandas"),
        requirement("scipy"),
        requirement("six"),
        requirement("tensorflow"),
        requirement("tensorflow_probability"),
    ],
)

build_test(
    name = "hierarchical_model_build_test",
    targets = [":hierarchical_model"],
)
