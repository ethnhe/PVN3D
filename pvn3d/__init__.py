from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

__version__ = "2.1.1"

try:
    __PVN3D_SETUP__
except NameError:
    __PVN3D_SETUP__ = False

if not __PVN3D_SETUP__:
    from pvn3d.lib import pointnet2_utils
    from pvn3d import data
    from pvn3d import models
