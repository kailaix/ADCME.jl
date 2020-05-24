if (WIN32)
add_library(winlu SHARED lru_cache.cpp Factorization/SparseFactorization.cpp Solve/Solve.cpp)
set_property(TARGET winlu PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(winlu ${TF_LIB_FILE})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set_target_properties(winlu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/build)
else()


add_library(lru SHARED lru_cache.cpp)
set_property(TARGET lru PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lru ${TF_LIB_FILE})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)


add_library(factorization SHARED Factorization/SparseFactorization.cpp)
set_property(TARGET factorization PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(factorization lru)
target_link_libraries(factorization ${TF_LIB_FILE} lru)
set_target_properties(factorization PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/build)


add_library(Solve SHARED Solve/Solve.cpp)
set_property(TARGET Solve PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(Solve lru)
target_link_libraries(Solve ${TF_LIB_FILE} lru)
set_target_properties(Solve PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/build)


endif()