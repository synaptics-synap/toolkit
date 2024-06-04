include(CMakeParseArguments)

# Variable used to serialize model build, since parallel execution is not supported in acuity
set(SYNAP_PREVIOUS_MODEL_TARGET "" CACHE INTERNAL "Previous model target")

function(synap_convert_all_models MODEL_CLASS)
    file(GLOB model_dirs */model)
    list(FILTER model_dirs EXCLUDE REGEX ".*_disabled")
    foreach(model_dir IN LISTS model_dirs)
        get_filename_component(BASE_DIR "${model_dir}" DIRECTORY)
        get_filename_component(CHALLANGE "${BASE_DIR}" NAME)
        file(GLOB MODELS ${model_dir}/*.tflite ${model_dir}/*/*.tflite
                         ${model_dir}/*.pb ${model_dir}/*/*.pb
                         ${model_dir}/*.prototxt ${model_dir}/*/*.prototxt
                         ${model_dir}/*.onnx ${model_dir}/*/*.onnx
                         ${model_dir}/*.torchscript ${model_dir}/*/*.torchscript)
        list(FILTER MODELS EXCLUDE REGEX ".*_float.tflite")
        list(FILTER MODELS EXCLUDE REGEX ".*_disabled.*")
        foreach(MODEL IN LISTS MODELS)
            synap_convert_models(MODEL ${MODEL} METAFILE "[=]" DEST ${MODEL_CLASS}/${CHALLANGE})
        endforeach()
        if(EXISTS "${BASE_DIR}/sample")
            synap_install_model_resources(RES "${BASE_DIR}/sample" DEST ${MODEL_CLASS}/${CHALLANGE})
        endif()
        if(EXISTS "${BASE_DIR}/info.json")
            synap_install_model_resources(RES "${BASE_DIR}/info.json" DEST ${MODEL_CLASS}/${CHALLANGE})
        endif()
    endforeach()
endfunction()


function(synap_convert_models)
    set(optional "")
    set(one MODEL METAFILE DEST)
    set(multiple "")
    set(SECURE_SOCS "VS640A0;VS680A0")
    cmake_parse_arguments(ARG "${optional}" "${one}" "${multiple}" "${ARGV}")
    get_metafile_paths("${ARG_MODEL}" "${ARG_METAFILE}" METAFILE_NAMES)
    foreach(METAFILE_NAME IN LISTS METAFILE_NAMES)
        get_filename_component(NAME "${METAFILE_NAME}" NAME_WLE)
        get_filename_component(METAFILE_PATH "${METAFILE_NAME}" DIRECTORY)
        if (NAME STREQUAL "model_metafile")
            get_filename_component(NAME "${MODEL}" NAME_WLE)
        endif()
        foreach(soc_it IN LISTS SOC)
            # Read file ${METAFILE_NAME}
            file(READ ${METAFILE_NAME} METAFILE_CONTENT)
            string(FIND "${METAFILE_CONTENT}" " secure:" IS_SECURE)
            if ((${IS_SECURE} EQUAL -1) AND (${soc_it} IN_LIST SECURE_SOCS))
                # message("${NAME} is NOT secure, skip for ${soc_it}")
                continue()
            endif()
            synap_convert_model(
                NAME ${NAME} MODEL ${ARG_MODEL} METAFILE ${METAFILE_NAME}
                SOC ${soc_it} DEST ${ARG_DEST}
            )
            install(FILES ${MODEL_BG} ${MODEL_META} ${MODEL_INFO} DESTINATION models/${soc_it}/${ARG_DEST}/model/${NAME})
            if(EXISTS "${METAFILE_PATH}/resources")
                install(DIRECTORY "${METAFILE_PATH}/resources" DESTINATION models/${soc_it}/${ARG_DEST}/model/${NAME})
            endif()
        endforeach()
    endforeach()
endfunction()

function(synap_convert_model)
    set(optional PROFILING)
    set(optional CPU_PROFILING)
    set(one NAME MODEL METAFILE SOC DEST)
    set(multiple "")
    cmake_parse_arguments(ARG "${optional}" "${one}" "${multiple}" "${ARGV}")
    set(NAME ${ARG_NAME})

    if ("${ARG_PROFILING}" OR ARG_METAFILE MATCHES "_profiling.yaml$")
        set(PROFILING "--profiling")
    endif()
    if(ENABLE_CPU_PROFILING)
        set(CPU_PROFILING "--cpu-profiling")
    endif()

    set(SYNAP_CONVERT "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/synap_convert.py")
    set(SYNAP_WORKDIR "${CMAKE_BINARY_DIR}/work")
    set(SYNAP_OUTDIR ${SYNAP_WORKDIR}/${ARG_SOC}/${ARG_DEST}/${NAME})
    get_filename_component(MODEL_EXTENSION "${ARG_MODEL}" LAST_EXT)
    if("${MODEL_EXTENSION}" STREQUAL ".prototxt")
        get_filename_component(MODEL_DIRECTORY "${ARG_MODEL}" DIRECTORY)
        get_filename_component(MODEL_NAME "${ARG_MODEL}" NAME_WLE)
        set(WEIGHTS_OPTION --weights)
        # Weights file must have extension ".caffemodel" in the same directory
        set(MODEL_WEIGHTS ${MODEL_DIRECTORY}/${MODEL_NAME}.caffemodel)
    endif()
    set(MODEL_BG ${SYNAP_OUTDIR}/model.synap)
    set(MODEL_META "")
    set(MODEL_INFO ${SYNAP_OUTDIR}/model_info.txt)
    set(MODEL_BG ${MODEL_BG} PARENT_SCOPE)
    set(MODEL_META ${MODEL_META} PARENT_SCOPE)
    set(MODEL_INFO ${MODEL_INFO} PARENT_SCOPE)
    set(CONVERSION_OPTIONS_FILE ${SYNAP_OUTDIR}/.conversion_options.yaml)

    configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/conversion_options.yaml.in ${CONVERSION_OPTIONS_FILE})

    set(TARGET_NAME ${ARG_DEST}_${NAME}_${ARG_SOC})
    string(REPLACE "/" "_" TARGET_NAME ${TARGET_NAME})
    string(REPLACE "@" "_" TARGET_NAME ${TARGET_NAME})
    add_custom_target(${TARGET_NAME} ALL DEPENDS ${MODEL_BG} ${MODEL_META})

    if(NOT "${ARG_METAFILE}" STREQUAL "")
        set(METAFILE_OPTION --meta)
    endif()
    set(SYNAP_PREVIOUS_MODEL_TARGET "${TARGET_NAME}" CACHE INTERNAL  "Previous model target")

    set(SYNAP_SECURITY_ENABLED false)
    if(ARG_SOC STREQUAL "VS640A0")
        set(CHIP_DIR "vs640/A0/")
        set(SYNAP_SECURITY_ENABLED true)
    elseif(ARG_SOC STREQUAL "VS680A0")
        set(CHIP_DIR "vs680/A0/")
        set(SYNAP_SECURITY_ENABLED true)
    endif()
    add_custom_command(
      OUTPUT ${MODEL_BG} ${MODEL_META} ${MODEL_INFO}
      DEPENDS ${ARG_MODEL} ${ARG_METAFILE} ${CONVERSION_OPTIONS_FILE}
      COMMAND SYNAP_SECURITY_ENABLED=${SYNAP_SECURITY_ENABLED}
              KEYS_BASE_DIR=${VSSDK_DIR}/security/keys/chip/${CHIP_DIR}
              CERTIFICATES_BASE_DIR=${VSSDK_DIR}/configs/product/common/${CHIP_DIR}
              ${SYNAP_CONVERT} ${PROFILING} --model ${ARG_MODEL} ${WEIGHTS_OPTION} ${MODEL_WEIGHTS} ${METAFILE_OPTION} ${ARG_METAFILE} --target ${ARG_SOC}
              --out-dir ${SYNAP_OUTDIR} --tools-dir ${VIP_SDK_DIR} --vssdk-dir ${VSSDK_DIR} ${CPU_PROFILING}
              --preserve
    )
endfunction()


function(get_metafile_paths MODEL META ACTUAL_METAFILE_NAME)
    get_filename_component(MODEL_NAME ${MODEL} NAME_WLE)
    get_filename_component(MODEL_PATH ${MODEL} DIRECTORY)
    if ("${META}" STREQUAL "[=]")
        # Use a metafile with the name same as the model but with different extension (if it exists)
        if(EXISTS "${MODEL_PATH}/${MODEL_NAME}.yaml")
            # Use companion .yaml file
            set(METAFILE_NAME "${MODEL_PATH}/${MODEL_NAME}.yaml")
        elseif(IS_DIRECTORY "${MODEL_PATH}/${MODEL_NAME}")
            # Use all .yaml files in companion directory
            file(GLOB METAFILE_NAME "${MODEL_PATH}/${MODEL_NAME}/*.yaml")
        else()
            # Check for default metafile in current and parent directories
            foreach(PREFIX_PATH IN ITEMS "." ".." "../.." "../../..")
                if(EXISTS "${MODEL_PATH}/${PREFIX_PATH}/model_metafile.yaml")
                    set(METAFILE_NAME "${MODEL_PATH}/${PREFIX_PATH}/model_metafile.yaml")
                    break()
                endif()
            endforeach()
        endif()
    else()
        set(METAFILE_NAME "${META}")
    endif()
    set (${ACTUAL_METAFILE_NAME} ${METAFILE_NAME} PARENT_SCOPE)
endfunction()


function(synap_install_model_resources)
    set(optional "")
    set(one RES DEST)
    set(multiple "")
    cmake_parse_arguments(ARG "${optional}" "${one}" "${multiple}" "${ARGV}")
    foreach(soc_it npuid_it IN ZIP_LISTS SOC NPUID)
        if(IS_DIRECTORY ${ARG_RES})
            install(DIRECTORY ${ARG_RES} DESTINATION models/${soc_it}/${ARG_DEST})
        else()
            install(FILES ${ARG_RES} DESTINATION models/${soc_it}/${ARG_DEST})
        endif()
    endforeach()
endfunction()
