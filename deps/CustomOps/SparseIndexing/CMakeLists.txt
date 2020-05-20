add_library(SparseIndexing SHARED SparseIndexing.cpp)
set_property(TARGET SparseIndexing PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SparseIndexing ${TF_LIB_FILE})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set_target_properties(SparseIndexing PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/build)

