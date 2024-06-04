How to run the tests
--------------------

The tests are executed in the builder docker, the builder docker shall have access to the following:

- SyNAP source tree (including external/vsi_acuity, models and framework)
- SyNAP host build with install
- VSSDK source tree
- VSSDK build for target

Assuming the above are available in the following locations:

- SyNAP source tree: ${TOPDIR}/synap
- SyNAP host build: ${TOPDIR}/synap/out
- SyNAP host build: ${TOPDIR}/synap-out
- VSSDK source tree: ${TOPDIR}/vssdk
- VSSDK build for target: ${TOPDIR}/vssdk/out

In this case the builder docker can be started as follows::

    cd ${TOPDIR}
    ${TOPDIR}/synap/scripts/builder.sh bash

The environment for the tests is configured as follows::

    export TOPDIR=<path to the root of the source tree>
    export TOOLS_DIR=${TOPDIR}/synap/out/install
    export VSSDK_DIR=${TOPDIR}/vssdk/
    export MODELS_DIR=${TOPDIR}/synap/models
    export PYTHONPATH=${TOPDIR}/synap/framework/toolkit:${TOPDIR}/synap/acuity

The tests can then be executed as follows:

    cd ${TOPDIR}/synap/framework/toolkit
    pytest

Some tests are by default excluded as they are slow. To run also these slow tests use the following command line::

    pytest --runslow

Single tests can be executed by specifying the relevant file name and tests can further be filtered with the ``-k``
option. For more information refer to the py-test documentation.

To generate the coverage the following command can be used:

    pytest --cov=pysynap  --cov-report html
