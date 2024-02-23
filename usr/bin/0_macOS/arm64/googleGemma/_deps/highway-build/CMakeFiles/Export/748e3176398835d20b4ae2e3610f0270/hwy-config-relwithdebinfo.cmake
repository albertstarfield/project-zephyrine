#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hwy::hwy" for configuration "RelWithDebInfo"
set_property(TARGET hwy::hwy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(hwy::hwy PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libhwy.a"
  )

list(APPEND _cmake_import_check_targets hwy::hwy )
list(APPEND _cmake_import_check_files_for_hwy::hwy "${_IMPORT_PREFIX}/lib/libhwy.a" )

# Import target "hwy::hwy_contrib" for configuration "RelWithDebInfo"
set_property(TARGET hwy::hwy_contrib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(hwy::hwy_contrib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libhwy_contrib.a"
  )

list(APPEND _cmake_import_check_targets hwy::hwy_contrib )
list(APPEND _cmake_import_check_files_for_hwy::hwy_contrib "${_IMPORT_PREFIX}/lib/libhwy_contrib.a" )

# Import target "hwy::hwy_test" for configuration "RelWithDebInfo"
set_property(TARGET hwy::hwy_test APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(hwy::hwy_test PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libhwy_test.a"
  )

list(APPEND _cmake_import_check_targets hwy::hwy_test )
list(APPEND _cmake_import_check_files_for_hwy::hwy_test "${_IMPORT_PREFIX}/lib/libhwy_test.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
