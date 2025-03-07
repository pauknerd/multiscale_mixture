# Set the name of the project and target:
SET(TARGET "MSHCMM")

# CMake version and CXX standard
CMAKE_MINIMUM_REQUIRED(VERSION 3.20.0)
set(CMAKE_CXX_STANDARD 17)

# search for deal.ii
FIND_PACKAGE(deal.II 9.6 QUIET
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# collect all src files
FILE(GLOB_RECURSE TARGET_SRC  "src/*.cpp")

# Set the include directory and the name of the project
INCLUDE_DIRECTORIES(include)
PROJECT(${TARGET})

# set library
ADD_LIBRARY(${TARGET}  ${TARGET_SRC})

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(${TARGET})

ADD_SUBDIRECTORY(apps)


#IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
#    SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
#ENDIF()

# documentation should be included
#ADD_SUBDIRECTORY(doc/doxygen)

# tests should be included
#enable_testing()
#ADD_SUBDIRECTORY(tests)

