cmake_minimum_required(VERSION 3.19)
project(StyleTransferNet LANGUAGES CXX)

# Use Qt6
cmake_policy(SET CMP0144 NEW)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets)

# OpenCV
set(OpenCV_DIR "D:/opencv/build")
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# LibTorch (CPU-only)
set(Torch_ROOT "D:/libtorch")
find_package(Torch REQUIRED PATHS ${Torch_ROOT} NO_DEFAULT_PATH)
message(STATUS "Using LibTorch (CPU-only): ${TORCH_LIBRARIES}")

# Source files
set(SRC_FILES
    main.cpp
    model.cpp
    model.h
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

# Executable
qt_add_executable(StyleTransferNet
    WIN32 MACOSX_BUNDLE
    ${SRC_FILES}
)

# Link libraries
target_link_libraries(StyleTransferNet
    PRIVATE
        Qt6::Widgets
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)

# RPATH for LibTorch DLLs/SOs
if (UNIX)
    set_target_properties(StyleTransferNet PROPERTIES
        BUILD_RPATH "${Torch_ROOT}/lib"
    )
endif()

# Copy model file to build directory
add_custom_command(TARGET StyleTransferNet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}
        $<TARGET_FILE_DIR:StyleTransferNet>
)

# Install rules
include(GNUInstallDirs)
install(TARGETS StyleTransferNet
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Deploy script
qt_generate_deploy_app_script(
    TARGET StyleTransferNet
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})

