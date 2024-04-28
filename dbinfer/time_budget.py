# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class TimeBudgetedIterator(object):
    def __init__(self, iter_, budget):
        self.iter_ = iter_
        self.budget = budget
        self.time_elapsed = 0

    def __iter__(self):
        self.time_elapsed = 0
        count = 1
        for item in self.iter_:
            t0 = time.time()
            yield item
            tt = time.time()
            self.time_elapsed += tt - t0
            average_iteration_time = self.time_elapsed / count

            if self.budget > 0 and average_iteration_time * (count + 1) > self.budget:
                logger.warning("Going to exceed time limit, terminating loop.")
                break
            count += 1

    def __len__(self):
        return len(self.iter_)
