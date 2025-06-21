-- File: src/zephyrine-c_curl.ads (DEFINITIVE VERSION)

with System;
with Interfaces.C;
with Zephyrine.C_Types;

package Zephyrine.C_Curl is
   use Zephyrine.C_Types;

   type CURL_Handle is new Opaque_Handle;
   type SList_Handle is new Opaque_Handle;

   type Write_Callback is access function
     (Contents   : System.Address;
      Size       : Interfaces.C.size_t;
      N_Members  : Interfaces.C.size_t;
      User_Data  : System.Address)
      return Interfaces.C.size_t;

   function curl_easy_init return CURL_Handle;
   pragma Import (C, curl_easy_init);

   procedure curl_easy_cleanup (Handle : CURL_Handle);
   pragma Import (C, curl_easy_cleanup);

   function curl_easy_perform (Handle : CURL_Handle) return Interfaces.C.int;
   pragma Import (C, curl_easy_perform);

   procedure curl_easy_setopt (Handle : CURL_Handle; Option : Interfaces.C.int; Value : System.Address);
   pragma Import (C, curl_easy_setopt, "curl_easy_setopt");

   procedure curl_easy_setopt_string (Handle : CURL_Handle; Option : Interfaces.C.int; Value : C_String);
   pragma Import (C, curl_easy_setopt_string, "curl_easy_setopt");

   function curl_slist_append (List : SList_Handle; Header : C_String) return SList_Handle;
   pragma Import (C, curl_slist_append);

   procedure curl_slist_free_all (List : SList_Handle);
   pragma Import (C, curl_slist_free_all);

   -- Common cURL options
   CURLOPT_URL           : constant Interfaces.C.int := 10002;
   CURLOPT_POSTFIELDS    : constant Interfaces.C.int := 10015;
   CURLOPT_WRITEFUNCTION : constant Interfaces.C.int := 20011;
   CURLOPT_WRITEDATA     : constant Interfaces.C.int := 10001;
   CURLOPT_HTTPHEADER    : constant Interfaces.C.int := 10023;
   CURLOPT_ERRORBUFFER   : constant Interfaces.C.int := 10010; -- <<< THE MISSING CONSTANT IS NOW HERE
end Zephyrine.C_Curl;