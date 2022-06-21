############  Set the path of the jobmodel library.
set(JOBMODEL_SRC ${IOANALYTICS_SRC}/jobmodel)

########### target to install the jobmodel librairy ############
add_custom_target(install-jobmodel-lib
    DEPENDS python-virt-env install-tools-lib install-extensions-lib
    WORKING_DIRECTORY ${JOBMODEL_SRC}
    COMMAND env ${ENV_PROXY} ${IOANALYTICS_RPM_VERSION} ${VIRT_PYTHON} -m pip install . ${PIP_INSTALL_OPT} ${PIP_OPT})

########### target to download the jobmodel dependencies ############
set(EXTERNAL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/external_deps.txt)
add_custom_command(OUTPUT ${EXTERNAL_DEPS}
    WORKING_DIRECTORY ${JOBMODEL_SRC}
    COMMAND grep -ivE ${LOCAL_LIBS} requirements.txt > ${EXTERNAL_DEPS} VERBATIM)
    
add_custom_target(download-jobmodel-deps
    DEPENDS create-deps-repo ${EXTERNAL_DEPS}
    WORKING_DIRECTORY ${JOBMODEL_SRC}
    COMMAND env ${ENV_PROXY} pip download -r ${EXTERNAL_DEPS} -d ${PY_DEPS_DIR} ${PIP_OPT} VERBATIM)

########### targets to byte-compile the jobmodel library ############
add_custom_target(compile-jobmodel
    WORKING_DIRECTORY ${IOANALYTICS_SRC_COPY}/jobmodel
    DEPENDS python-src-copy
    COMMAND python -m compileall -b -f .
    COMMENT echo "Byte-compiling all python modules present in the jobmodel source dir")

add_custom_target(select-compiled-jobmodel
    WORKING_DIRECTORY ${IOANALYTICS_SRC_COPY}/jobmodel
    DEPENDS compile-jobmodel
    COMMAND find . -type f -name \"*.py\" ! -name \"__init__.py\" ! -name \"setup.py\" -delete
    COMMENT "Remove all jobmodel sources .py files except for __init__ file which are required for packaging")

add_subdirectory(tests)
