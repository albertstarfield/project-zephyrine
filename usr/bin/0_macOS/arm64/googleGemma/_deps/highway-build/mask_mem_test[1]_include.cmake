if(EXISTS "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/highway-build/mask_mem_test[1]_tests.cmake")
  include("/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/highway-build/mask_mem_test[1]_tests.cmake")
else()
  add_test(mask_mem_test_NOT_BUILT mask_mem_test_NOT_BUILT)
endif()