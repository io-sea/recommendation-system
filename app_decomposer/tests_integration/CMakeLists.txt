########### target to install the test requirements ############
add_custom_target(install-tools-int-test-lib
	DEPENDS install-tools-lib
	WORKING_DIRECTORY ${TOOLS_SRC}
	COMMAND env ${ENV_PROXY} ${IOANALYTICS_RPM_VERSION} ${VIRT_PYTHON} -m pip install .[test-int] ${PIP_OPT})

########### add a "python-tools-int" target to test python code and, if "coverage" is available, generate report on code coverage
if(USE_INTEGRATION_TESTS)
	find_package(PyCoverage)

	if(PYCOVERAGE_EXEC AND USE_COVERAGE)
		add_custom_target(python-tools-int
			WORKING_DIRECTORY ${TOOLS_SRC}
			DEPENDS install-tools-lib install-tools-int-test-lib deploy-integration-platform-ioanalyticstools
			COMMAND ${VIRT_PYTHON} -m coverage run -m unittest discover -v tests_integration
			COMMAND ${VIRT_PYTHON} -m coverage xml --include="${TOOLS_SRC}/*" --omit="*/tests/*,*/tests_integration/*,*/__init__.py" -o ${REPORTS_DIR}/pycoverage-tools-int.xml
			COMMAND ${VIRT_PYTHON} -m coverage report --include="${TOOLS_SRC}/*" --omit="*/tests*,*/tests_integration/*,*/__init__.py"
			COMMENT "Running Python integration tests for ioanalyticstools library and generating code coverage report")
	else(PYCOVERAGE_EXEC AND USE_COVERAGE)
		add_custom_target(python-tools-int
			WORKING_DIRECTORY ${TOOLS_SRC}
			DEPENDS install-tools-lib install-tools-int-test-lib deploy-integration-platform-ioanalyticstools
			COMMAND ${VIRT_PYTHON} -m unittest discover -v -s tests_integration
			COMMENT "Running Python integration tests for ioanalyticstools library")
	endif(PYCOVERAGE_EXEC AND USE_COVERAGE)

    add_custom_target(deploy-integration-platform-ioanalyticstools
        WORKING_DIRECTORY ${INTEGRATION_PLATFORM_SRC}
        COMMAND ansible-playbook -i hosts.ini deploy.yml --extra-vars "rhel_distrib=${RHEL_DISTRIB}" --extra-vars "docker_ioi_image_tag=${COMPVERS}" --extra-vars "docker_keycloak_image_tag=${COMPVERS}"
        COMMAND ansible-playbook -i files/iopa-integration.ini populate_mongodb.yml --extra-vars "@${CMAKE_CURRENT_SOURCE_DIR}/iopa_integration_config.json"
        COMMAND ansible-playbook -i files/iopa-integration.ini "${CMAKE_CURRENT_SOURCE_DIR}/ioanalyticstools_deploy.yml"
        COMMENT "Preparing iopa integration platform for ioanalyticstools integration tests")

    add_custom_command(TARGET python-tools-int
        POST_BUILD
        WORKING_DIRECTORY ${INTEGRATION_PLATFORM_SRC}
        COMMAND ansible-playbook -i hosts.ini undeploy.yml --extra-vars "rhel_distrib=${RHEL_DISTRIB}" --extra-vars "docker_ioi_image_tag=${COMPVERS}" --extra-vars "docker_keycloak_image_tag=${COMPVERS}"
        COMMENT "Undeploying iopa integration platform")
endif(USE_INTEGRATION_TESTS)
