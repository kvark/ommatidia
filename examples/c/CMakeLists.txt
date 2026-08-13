cmake_minimum_required(VERSION 3.20)
project(ommatidia_c_example C)

set(OMMATIDIA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH
    "Ommatidia source or installed SDK root")
set(OMMATIDIA_LIBRARY "" CACHE FILEPATH
    "Path to libommatidia_native (required for an installed SDK)")

add_executable(ommatidia-inspect inspect.c)
target_include_directories(ommatidia-inspect PRIVATE "${OMMATIDIA_ROOT}/include")
if(OMMATIDIA_LIBRARY)
    target_link_libraries(ommatidia-inspect PRIVATE "${OMMATIDIA_LIBRARY}")
else()
    find_library(OMMATIDIA_LIBRARY_AUTO ommatidia_native
        PATHS "${OMMATIDIA_ROOT}/target/release" REQUIRED NO_DEFAULT_PATH)
    target_link_libraries(ommatidia-inspect PRIVATE "${OMMATIDIA_LIBRARY_AUTO}")
endif()
