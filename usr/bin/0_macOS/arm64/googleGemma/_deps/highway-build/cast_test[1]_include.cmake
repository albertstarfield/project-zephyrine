if(EXISTS "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/highway-build/cast_test[1]_tests.cmake")
  include("/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/highway-build/cast_test[1]_tests.cmake")
else()
  add_test(cast_test_NOT_BUILT cast_test_NOT_BUILT)
endif()