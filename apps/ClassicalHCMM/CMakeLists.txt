set(TARGET "Maes_Cylinder_Test")
add_executable(${TARGET} Maes_cylinder_test.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Maes_Cube_Test")
add_executable(${TARGET} Maes_cube_test_cauchy_stress.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Maes_Cube_Test_DirichletBC")
add_executable(${TARGET} Maes_cube_test_dirichletBC.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Braeu_Cylinder_Test")
add_executable(${TARGET} Braeu_cylinder_test.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Latorre_Cylinder_Test")
add_executable(${TARGET} Latorre_cylinder_test.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)
