if(EXISTS "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/highway-build/blockwise_shift_test[1]_tests.cmake")
  include("/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/highway-build/blockwise_shift_test[1]_tests.cmake")
else()
  add_test(blockwise_shift_test_NOT_BUILT blockwise_shift_test_NOT_BUILT)
endif()