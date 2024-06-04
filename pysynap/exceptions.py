# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

class SynapError(Exception):
    pass

class ConversionError(SynapError):
    pass

class ProfilingError(SynapError):
    pass


