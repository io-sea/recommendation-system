########### target to install the test requirements ############
add_custom_target(install-jobmodel-test-lib
	DEPENDS install-jobmodel-lib
	WORKING_DIRECTORY ${JOBMODEL_SRC}
	COMMAND env ${ENV_PROXY} ${IOANALYTICS_RPM_VERSION} ${VIRT_PYTHON} -m pip install .[test] ${PIP_OPT})

########### add a "python-jobmodel-ut" target to test python code and, if "coverage" is available, generate report on code coverage
if(USE_UNIT_TESTS)
    find_package(PyCoverage)
    if(PYCOVERAGE_EXEC AND USE_COVERAGE)
        add_custom_target(python-jobmodel-ut
            WORKING_DIRECTORY ${JOBMODEL_SRC}
            DEPENDS install-jobmodel-test-lib
            COMMAND ${VIRT_PYTHON} -m coverage run -m unittest -v
            COMMAND ${VIRT_PYTHON} -m coverage xml --include="${JOBMODEL_SRC}/*" --omit="*/tests/*,*/__init__.py"  -o ${REPORTS_DIR}/pycoverage-jobmodel.xml
            COMMAND ${VIRT_PYTHON} -m coverage report --include="${JOBMODEL_SRC}/*" --omit="*/tests/test*,*/__init__.py"
            COMMENT "Running Python unit tests for jobmodel library and generating code coverage report")
    else(PYCOVERAGE_EXEC AND USE_COVERAGE)
        add_custom_target(python-jobmodel-ut
            WORKING_DIRECTORY ${JOBMODEL_SRC}
            DEPENDS install-jobmodel-test-lib
            COMMAND ${VIRT_PYTHON} -m unittest -v
            COMMENT "Running Python unit tests for jobmodel library")
    endif(PYCOVERAGE_EXEC AND USE_COVERAGE)
endif(USE_UNIT_TESTS)
