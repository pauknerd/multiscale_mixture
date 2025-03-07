set(TARGET "Maes_Cylinder_Test_Prestretch")
add_executable(${TARGET} Maes_cylinder_test_prestretch.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Maes_Cube_Test_Prestretch")
add_executable(${TARGET} Maes_cube_test_prestretch_cauchy_stress.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Braeu_Cylinder_Test_Prestretch")
add_executable(${TARGET} Braeu_cylinder_test_prestretch.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Latorre_Cylinder_Test_Prestretch")
add_executable(${TARGET} Latorre_cylinder_test_prestretch.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Latorre_Cylinder_Test_Prestretch_ActiveStress")
add_executable(${TARGET} Latorre_cylinder_test_prestretch_incl_active_stress.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)