add_library(SparseAccumulator SHARED SparseAccumulator.cpp Impl.cpp)
set_property(TARGET SparseAccumulator PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SparseAccumulator ${TF_LIB_FILE})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set_target_properties(SparseAccumulator PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_CURRENT_SOURCE_DIR}/build)

