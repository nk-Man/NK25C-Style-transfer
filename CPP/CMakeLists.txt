cmake_minimum_required(VERSION 3.19)
project(StyleTransferNet LANGUAGES CXX)

# Use Qt6
cmake_policy(SET CMP0144 NEW)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt6
find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets)

# Find OpenCV
set(OpenCV_DIR "D:/opencv/build")
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# Find LibTorch (CPU-only)
set(Torch_ROOT "D:/libtorch")
find_package(Torch REQUIRED PATHS ${Torch_ROOT} NO_DEFAULT_PATH)
message(STATUS "Using LibTorch (CPU-only): ${TORCH_LIBRARIES}")

# Include directories for OpenCV and Torch headers
target_include_directories(StyleTransferNet PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Source files
set(SRC_FILES
    main.cpp
    model.cpp
    model.h
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

# Create executable with Qt support
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

# Copy model file to build directory (adjust path to your serialized model)
add_custom_command(TARGET StyleTransferNet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/model.pt
        $<TARGET_FILE_DIR:StyleTransferNet>
)

# Install rules
include(GNUInstallDirs)
install(TARGETS StyleTransferNet
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Deploy script for Qt
qt_generate_deploy_app_script(
    TARGET StyleTransferNet
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
