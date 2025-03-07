
set(TARGET "Latorre_Coupled_Cylinder_Test_Simple_Pathways")
add_executable(${TARGET} Latorre_coupled_cylinder_test_simple_pathways.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Latorre_Coupled_Cylinder_Test")
add_executable(${TARGET} Latorre_coupled_cylinder_test.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Latorre_Coupled_Cylinder_Test_Split_EC")
add_executable(${TARGET} Latorre_coupled_cylinder_test_split_EC.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)


set(TARGET "Latorre_Coupled_Cylinder_Test_Split_EC_Active_Stress")
add_executable(${TARGET} Latorre_coupled_cylinder_test_split_EC_active_stress.cpp)
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} MSHCMM)
