add_test([=[NanobenchmarkTest.RunAll]=]  /Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build/tests/nanobenchmark_test [==[--gtest_filter=NanobenchmarkTest.RunAll]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[NanobenchmarkTest.RunAll]=]  PROPERTIES WORKING_DIRECTORY /Users/user/Documents/misc/AI/LLM/alpaca-electron-zephyrine/usr/vendor/gemma.cpp/build/_deps/highway-build SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  nanobenchmark_test_TESTS NanobenchmarkTest.RunAll)
