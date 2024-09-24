//
// Copyright (c) 2019-2024 Ruben Perez Hidalgo (rubenperez038 at gmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_MYSQL_TEST_INTEGRATION_INCLUDE_TEST_INTEGRATION_NETWORK_TEST_HPP
#define BOOST_MYSQL_TEST_INTEGRATION_INCLUDE_TEST_INTEGRATION_NETWORK_TEST_HPP

#include <boost/test/unit_test.hpp>

#include <iterator>

#include "test_common/stringize.hpp"

/**
 * Defines the required infrastructure for network tests.
 * These are data-driven (parametrized) test cases. We don't
 * use Boost.Test data-driven functionality for this because
 * tests must have different labels depending on the parameters,
 * which is not supported by Boost.Test.
 *
 * All network tests employ type-erased network objects,
 * defined under the utils/ folder. Network tests are run under
 * different network variants. Each variant is a combination of a Stream
 * and a sync/async flavor. Examples: TCP + async with callbacks,
 * UNIX SSL + sync with exceptions, TCP SSL + C++20 coroutines.
 *
 * Access this feature using the BOOST_MYSQL_NETWORK_TEST_* macros;
 * they work similar to BOOST_AUTO_TEST_CASE.
 */

namespace boost {
namespace mysql {
namespace test {

// The type of a sample generated by DataGenerator
template <class SampleCollection>
using sample_type = typename std::decay<decltype(*std::begin(std::declval<const SampleCollection&>()))>::type;

inline boost::unit_test::test_suite* create_test_suite(
    boost::unit_test::const_string tc_name,
    boost::unit_test::const_string tc_file,
    std::size_t tc_line
)
{
    // Create a test suite with the name of the test
    auto* suite = new boost::unit_test::test_suite(tc_name, tc_file, tc_line);
    boost::unit_test::framework::current_auto_test_suite().add(suite);

    // Add decorators
    auto& collector = boost::unit_test::decorator::collector_t::instance();
    collector.store_in(*suite);
    collector.reset();

    return suite;
}

// Inspired in how Boost.Test auto-registers unit tests.
// BOOST_MYSQL_NETWORK_TEST defines a static variable of this
// type, which takes care of test registration.
template <class Testcase>
struct network_test_registrar
{
    template <class SampleCollection>
    network_test_registrar(
        boost::unit_test::const_string tc_name,
        boost::unit_test::const_string tc_file,
        std::size_t tc_line,
        const SampleCollection& samples
    )
    {
        // Create suite
        auto* suite = create_test_suite(tc_name, tc_file, tc_line);

        // Create a test for each sample
        for (const auto& sample : samples)
        {
            std::string test_name = stringize(sample);
            auto* test = boost::unit_test::make_test_case(
                [sample] {
                    Testcase tc_struct;
                    tc_struct.test_method(sample);
                },
                test_name,
                tc_file,
                tc_line
            );
            sample.set_test_attributes(*test);
            suite->add(test);
        }
    }
};

}  // namespace test
}  // namespace mysql
}  // namespace boost

#define BOOST_MYSQL_NETWORK_TEST(name, fixture, samples)                       \
    struct name : public fixture                                               \
    {                                                                          \
        void test_method(const sample_type<decltype(samples)>&);               \
    };                                                                         \
    static ::boost::mysql::test::network_test_registrar<name> name##_registrar \
        BOOST_ATTRIBUTE_UNUSED(#name, __FILE__, __LINE__, samples);            \
    void name::test_method(const sample_type<decltype(samples)>& sample)

#endif