/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 * SPDX-License-Identifier: curl
 *
 ***************************************************************************/
#include "curlcheck.h"
#include "netrc.h"
#include "memdebug.h" /* LAST include file */

#ifndef CURL_DISABLE_NETRC

static char *s_login;
static char *s_password;

static CURLcode unit_setup(void)
{
  s_password = strdup("");
  s_login = strdup("");
  if(!s_password || !s_login) {
    Curl_safefree(s_password);
    Curl_safefree(s_login);
    return CURLE_OUT_OF_MEMORY;
  }
  return CURLE_OK;
}

static void unit_stop(void)
{
  Curl_safefree(s_password);
  Curl_safefree(s_login);
}

UNITTEST_START
  int result;

  /*
   * Test a non existent host in our netrc file.
   */
  result = Curl_parsenetrc("test.example.com", &s_login, &s_password, arg);
  fail_unless(result == 1, "Host not found should return 1");
  abort_unless(s_password != NULL, "returned NULL!");
  fail_unless(s_password[0] == 0, "password should not have been changed");
  abort_unless(s_login != NULL, "returned NULL!");
  fail_unless(s_login[0] == 0, "login should not have been changed");

  /*
   * Test a non existent login in our netrc file.
   */
  free(s_login);
  s_login = strdup("me");
  abort_unless(s_login != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &s_login, &s_password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(s_password != NULL, "returned NULL!");
  fail_unless(s_password[0] == 0, "password should not have been changed");
  abort_unless(s_login != NULL, "returned NULL!");
  fail_unless(strncmp(s_login, "me", 2) == 0,
              "login should not have been changed");

  /*
   * Test a non existent login and host in our netrc file.
   */
  free(s_login);
  s_login = strdup("me");
  abort_unless(s_login != NULL, "returned NULL!");
  result = Curl_parsenetrc("test.example.com", &s_login, &s_password, arg);
  fail_unless(result == 1, "Host not found should return 1");
  abort_unless(s_password != NULL, "returned NULL!");
  fail_unless(s_password[0] == 0, "password should not have been changed");
  abort_unless(s_login != NULL, "returned NULL!");
  fail_unless(strncmp(s_login, "me", 2) == 0,
              "login should not have been changed");

  /*
   * Test a non existent login (substring of an existing one) in our
   * netrc file.
   */
  free(s_login);
  s_login = strdup("admi");
  abort_unless(s_login != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &s_login, &s_password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(s_password != NULL, "returned NULL!");
  fail_unless(s_password[0] == 0, "password should not have been changed");
  abort_unless(s_login != NULL, "returned NULL!");
  fail_unless(strncmp(s_login, "admi", 4) == 0,
              "login should not have been changed");

  /*
   * Test a non existent login (superstring of an existing one)
   * in our netrc file.
   */
  free(s_login);
  s_login = strdup("adminn");
  abort_unless(s_login != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &s_login, &s_password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(s_password != NULL, "returned NULL!");
  fail_unless(s_password[0] == 0, "password should not have been changed");
  abort_unless(s_login != NULL, "returned NULL!");
  fail_unless(strncmp(s_login, "adminn", 6) == 0,
              "login should not have been changed");

  /*
   * Test for the first existing host in our netrc file
   * with s_login[0] = 0.
   */
  free(s_login);
  s_login = strdup("");
  abort_unless(s_login != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &s_login, &s_password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(s_password != NULL, "returned NULL!");
  fail_unless(strncmp(s_password, "passwd", 6) == 0,
              "password should be 'passwd'");
  abort_unless(s_login != NULL, "returned NULL!");
  fail_unless(strncmp(s_login, "admin", 5) == 0, "login should be 'admin'");

  /*
   * Test for the first existing host in our netrc file
   * with s_login[0] != 0.
   */
  free(s_password);
  s_password = strdup("");
  abort_unless(s_password != NULL, "returned NULL!");
  result = Curl_parsenetrc("example.com", &s_login, &s_password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(s_password != NULL, "returned NULL!");
  fail_unless(strncmp(s_password, "passwd", 6) == 0,
              "password should be 'passwd'");
  abort_unless(s_login != NULL, "returned NULL!");
  fail_unless(strncmp(s_login, "admin", 5) == 0, "login should be 'admin'");

  /*
   * Test for the second existing host in our netrc file
   * with s_login[0] = 0.
   */
  free(s_password);
  s_password = strdup("");
  abort_unless(s_password != NULL, "returned NULL!");
  free(s_login);
  s_login = strdup("");
  abort_unless(s_login != NULL, "returned NULL!");
  result = Curl_parsenetrc("curl.example.com", &s_login, &s_password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(s_password != NULL, "returned NULL!");
  fail_unless(strncmp(s_password, "none", 4) == 0,
              "password should be 'none'");
  abort_unless(s_login != NULL, "returned NULL!");
  fail_unless(strncmp(s_login, "none", 4) == 0, "login should be 'none'");

  /*
   * Test for the second existing host in our netrc file
   * with s_login[0] != 0.
   */
  free(s_password);
  s_password = strdup("");
  abort_unless(s_password != NULL, "returned NULL!");
  result = Curl_parsenetrc("curl.example.com", &s_login, &s_password, arg);
  fail_unless(result == 0, "Host should have been found");
  abort_unless(s_password != NULL, "returned NULL!");
  fail_unless(strncmp(s_password, "none", 4) == 0,
              "password should be 'none'");
  abort_unless(s_login != NULL, "returned NULL!");
  fail_unless(strncmp(s_login, "none", 4) == 0, "login should be 'none'");

UNITTEST_STOP

#else
static CURLcode unit_setup(void)
{
  return CURLE_OK;
}
static void unit_stop(void)
{
}
UNITTEST_START
UNITTEST_STOP

#endif