#find_library(MGL mgl PATHS /usr/local/lib)

ADD_EXECUTABLE( easy-eye-tracker main.cpp eyeTracker.cpp eyeTracker.h globalVariables.cpp globalVariables.h findEyeCenter.cpp findEyeCenter.h findEyeCorner.cpp findEyeCorner.h constants.h helpers.cpp helpers.h)
find_package(X11 REQUIRED)
include_directories(${X11_INCLUDE_DIR})
TARGET_LINK_LIBRARIES( easy-eye-tracker ${OpenCV_LIBS} ${X11_LIBRARIES} )
#target_link_libraries( easy-eye-tracker ${MGL})
