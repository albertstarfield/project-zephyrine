pragma Warnings (Off);
pragma Ada_95;
with System;
with System.Parameters;
with System.Secondary_Stack;
package ada_main is

   gnat_argc : Integer;
   gnat_argv : System.Address;
   gnat_envp : System.Address;

   pragma Import (C, gnat_argc);
   pragma Import (C, gnat_argv);
   pragma Import (C, gnat_envp);

   gnat_exit_status : Integer;
   pragma Import (C, gnat_exit_status);

   GNAT_Version : constant String :=
                    "GNAT Version: 15.0.1 20250418 (prerelease)" & ASCII.NUL;
   pragma Export (C, GNAT_Version, "__gnat_version");

   GNAT_Version_Address : constant System.Address := GNAT_Version'Address;
   pragma Export (C, GNAT_Version_Address, "__gnat_version_address");

   Ada_Main_Program_Name : constant String := "_ada_ada_hellosekai" & ASCII.NUL;
   pragma Export (C, Ada_Main_Program_Name, "__gnat_ada_main_program_name");

   procedure adainit;
   pragma Export (C, adainit, "adainit");

   procedure adafinal;
   pragma Export (C, adafinal, "adafinal");

   function main
     (argc : Integer;
      argv : System.Address;
      envp : System.Address)
      return Integer;
   pragma Export (C, main, "main");

   type Version_32 is mod 2 ** 32;
   u00001 : constant Version_32 := 16#60f04058#;
   pragma Export (C, u00001, "ada_hellosekaiB");
   u00002 : constant Version_32 := 16#b2cfab41#;
   pragma Export (C, u00002, "system__standard_libraryB");
   u00003 : constant Version_32 := 16#6278fccd#;
   pragma Export (C, u00003, "system__standard_libraryS");
   u00004 : constant Version_32 := 16#76789da1#;
   pragma Export (C, u00004, "adaS");
   u00005 : constant Version_32 := 16#a201b8c5#;
   pragma Export (C, u00005, "ada__strings__text_buffersB");
   u00006 : constant Version_32 := 16#a7cfd09b#;
   pragma Export (C, u00006, "ada__strings__text_buffersS");
   u00007 : constant Version_32 := 16#e6d4fa36#;
   pragma Export (C, u00007, "ada__stringsS");
   u00008 : constant Version_32 := 16#70765b54#;
   pragma Export (C, u00008, "systemS");
   u00009 : constant Version_32 := 16#45e1965e#;
   pragma Export (C, u00009, "system__exception_tableB");
   u00010 : constant Version_32 := 16#fd5d2d4d#;
   pragma Export (C, u00010, "system__exception_tableS");
   u00011 : constant Version_32 := 16#7fa0a598#;
   pragma Export (C, u00011, "system__soft_linksB");
   u00012 : constant Version_32 := 16#a3fdee7d#;
   pragma Export (C, u00012, "system__soft_linksS");
   u00013 : constant Version_32 := 16#d0b087d0#;
   pragma Export (C, u00013, "system__secondary_stackB");
   u00014 : constant Version_32 := 16#debd0a58#;
   pragma Export (C, u00014, "system__secondary_stackS");
   u00015 : constant Version_32 := 16#33a162cd#;
   pragma Export (C, u00015, "ada__exceptionsB");
   u00016 : constant Version_32 := 16#00870947#;
   pragma Export (C, u00016, "ada__exceptionsS");
   u00017 : constant Version_32 := 16#85bf25f7#;
   pragma Export (C, u00017, "ada__exceptions__last_chance_handlerB");
   u00018 : constant Version_32 := 16#a028f72d#;
   pragma Export (C, u00018, "ada__exceptions__last_chance_handlerS");
   u00019 : constant Version_32 := 16#42d3e466#;
   pragma Export (C, u00019, "system__exceptionsS");
   u00020 : constant Version_32 := 16#c367aa24#;
   pragma Export (C, u00020, "system__exceptions__machineB");
   u00021 : constant Version_32 := 16#ec13924a#;
   pragma Export (C, u00021, "system__exceptions__machineS");
   u00022 : constant Version_32 := 16#7706238d#;
   pragma Export (C, u00022, "system__exceptions_debugB");
   u00023 : constant Version_32 := 16#40780307#;
   pragma Export (C, u00023, "system__exceptions_debugS");
   u00024 : constant Version_32 := 16#52e91815#;
   pragma Export (C, u00024, "system__img_intS");
   u00025 : constant Version_32 := 16#f2c63a02#;
   pragma Export (C, u00025, "ada__numericsS");
   u00026 : constant Version_32 := 16#174f5472#;
   pragma Export (C, u00026, "ada__numerics__big_numbersS");
   u00027 : constant Version_32 := 16#8a5c240d#;
   pragma Export (C, u00027, "system__unsigned_typesS");
   u00028 : constant Version_32 := 16#bca88fbc#;
   pragma Export (C, u00028, "system__storage_elementsS");
   u00029 : constant Version_32 := 16#5c7d9c20#;
   pragma Export (C, u00029, "system__tracebackB");
   u00030 : constant Version_32 := 16#f6ecafe9#;
   pragma Export (C, u00030, "system__tracebackS");
   u00031 : constant Version_32 := 16#5f6b6486#;
   pragma Export (C, u00031, "system__traceback_entriesB");
   u00032 : constant Version_32 := 16#b86ae4d8#;
   pragma Export (C, u00032, "system__traceback_entriesS");
   u00033 : constant Version_32 := 16#65d5266b#;
   pragma Export (C, u00033, "system__traceback__symbolicB");
   u00034 : constant Version_32 := 16#140ceb78#;
   pragma Export (C, u00034, "system__traceback__symbolicS");
   u00035 : constant Version_32 := 16#701f9d88#;
   pragma Export (C, u00035, "ada__exceptions__tracebackB");
   u00036 : constant Version_32 := 16#26ed0985#;
   pragma Export (C, u00036, "ada__exceptions__tracebackS");
   u00037 : constant Version_32 := 16#f9910acc#;
   pragma Export (C, u00037, "system__address_imageB");
   u00038 : constant Version_32 := 16#d19ac66e#;
   pragma Export (C, u00038, "system__address_imageS");
   u00039 : constant Version_32 := 16#45c8b1f1#;
   pragma Export (C, u00039, "system__img_address_32S");
   u00040 : constant Version_32 := 16#9111f9c1#;
   pragma Export (C, u00040, "interfacesS");
   u00041 : constant Version_32 := 16#68e81073#;
   pragma Export (C, u00041, "system__img_address_64S");
   u00042 : constant Version_32 := 16#fd158a37#;
   pragma Export (C, u00042, "system__wch_conB");
   u00043 : constant Version_32 := 16#a9757837#;
   pragma Export (C, u00043, "system__wch_conS");
   u00044 : constant Version_32 := 16#5c289972#;
   pragma Export (C, u00044, "system__wch_stwB");
   u00045 : constant Version_32 := 16#84645436#;
   pragma Export (C, u00045, "system__wch_stwS");
   u00046 : constant Version_32 := 16#7cd63de5#;
   pragma Export (C, u00046, "system__wch_cnvB");
   u00047 : constant Version_32 := 16#afb5b247#;
   pragma Export (C, u00047, "system__wch_cnvS");
   u00048 : constant Version_32 := 16#e538de43#;
   pragma Export (C, u00048, "system__wch_jisB");
   u00049 : constant Version_32 := 16#1a02d06d#;
   pragma Export (C, u00049, "system__wch_jisS");
   u00050 : constant Version_32 := 16#a43efea2#;
   pragma Export (C, u00050, "system__parametersB");
   u00051 : constant Version_32 := 16#45e1a745#;
   pragma Export (C, u00051, "system__parametersS");
   u00052 : constant Version_32 := 16#0286ce9f#;
   pragma Export (C, u00052, "system__soft_links__initializeB");
   u00053 : constant Version_32 := 16#ac2e8b53#;
   pragma Export (C, u00053, "system__soft_links__initializeS");
   u00054 : constant Version_32 := 16#8599b27b#;
   pragma Export (C, u00054, "system__stack_checkingB");
   u00055 : constant Version_32 := 16#b7294e42#;
   pragma Export (C, u00055, "system__stack_checkingS");
   u00056 : constant Version_32 := 16#8b7604c4#;
   pragma Export (C, u00056, "ada__strings__utf_encodingB");
   u00057 : constant Version_32 := 16#c9e86997#;
   pragma Export (C, u00057, "ada__strings__utf_encodingS");
   u00058 : constant Version_32 := 16#bb780f45#;
   pragma Export (C, u00058, "ada__strings__utf_encoding__stringsB");
   u00059 : constant Version_32 := 16#b85ff4b6#;
   pragma Export (C, u00059, "ada__strings__utf_encoding__stringsS");
   u00060 : constant Version_32 := 16#d1d1ed0b#;
   pragma Export (C, u00060, "ada__strings__utf_encoding__wide_stringsB");
   u00061 : constant Version_32 := 16#5678478f#;
   pragma Export (C, u00061, "ada__strings__utf_encoding__wide_stringsS");
   u00062 : constant Version_32 := 16#c2b98963#;
   pragma Export (C, u00062, "ada__strings__utf_encoding__wide_wide_stringsB");
   u00063 : constant Version_32 := 16#d7af3358#;
   pragma Export (C, u00063, "ada__strings__utf_encoding__wide_wide_stringsS");
   u00064 : constant Version_32 := 16#683e3bb7#;
   pragma Export (C, u00064, "ada__tagsB");
   u00065 : constant Version_32 := 16#4ff764f3#;
   pragma Export (C, u00065, "ada__tagsS");
   u00066 : constant Version_32 := 16#3548d972#;
   pragma Export (C, u00066, "system__htableB");
   u00067 : constant Version_32 := 16#f1af03bf#;
   pragma Export (C, u00067, "system__htableS");
   u00068 : constant Version_32 := 16#1f1abe38#;
   pragma Export (C, u00068, "system__string_hashB");
   u00069 : constant Version_32 := 16#56ea83c0#;
   pragma Export (C, u00069, "system__string_hashS");
   u00070 : constant Version_32 := 16#e7d0da5b#;
   pragma Export (C, u00070, "system__val_lluS");
   u00071 : constant Version_32 := 16#238798c9#;
   pragma Export (C, u00071, "system__sparkS");
   u00072 : constant Version_32 := 16#a571a4dc#;
   pragma Export (C, u00072, "system__spark__cut_operationsB");
   u00073 : constant Version_32 := 16#629c0fb7#;
   pragma Export (C, u00073, "system__spark__cut_operationsS");
   u00074 : constant Version_32 := 16#365e21c1#;
   pragma Export (C, u00074, "system__val_utilB");
   u00075 : constant Version_32 := 16#f3b10aca#;
   pragma Export (C, u00075, "system__val_utilS");
   u00076 : constant Version_32 := 16#b98923bf#;
   pragma Export (C, u00076, "system__case_utilB");
   u00077 : constant Version_32 := 16#bf658c01#;
   pragma Export (C, u00077, "system__case_utilS");
   u00078 : constant Version_32 := 16#27ac21ac#;
   pragma Export (C, u00078, "ada__text_ioB");
   u00079 : constant Version_32 := 16#60f53344#;
   pragma Export (C, u00079, "ada__text_ioS");
   u00080 : constant Version_32 := 16#b228eb1e#;
   pragma Export (C, u00080, "ada__streamsB");
   u00081 : constant Version_32 := 16#613fe11c#;
   pragma Export (C, u00081, "ada__streamsS");
   u00082 : constant Version_32 := 16#367911c4#;
   pragma Export (C, u00082, "ada__io_exceptionsS");
   u00083 : constant Version_32 := 16#05222263#;
   pragma Export (C, u00083, "system__put_imagesB");
   u00084 : constant Version_32 := 16#6cd85c4b#;
   pragma Export (C, u00084, "system__put_imagesS");
   u00085 : constant Version_32 := 16#22b9eb9f#;
   pragma Export (C, u00085, "ada__strings__text_buffers__utilsB");
   u00086 : constant Version_32 := 16#89062ac3#;
   pragma Export (C, u00086, "ada__strings__text_buffers__utilsS");
   u00087 : constant Version_32 := 16#1cacf006#;
   pragma Export (C, u00087, "interfaces__c_streamsB");
   u00088 : constant Version_32 := 16#d07279c2#;
   pragma Export (C, u00088, "interfaces__c_streamsS");
   u00089 : constant Version_32 := 16#fb523cdb#;
   pragma Export (C, u00089, "system__crtlS");
   u00090 : constant Version_32 := 16#ec2f4d1e#;
   pragma Export (C, u00090, "system__file_ioB");
   u00091 : constant Version_32 := 16#16390e12#;
   pragma Export (C, u00091, "system__file_ioS");
   u00092 : constant Version_32 := 16#c34b231e#;
   pragma Export (C, u00092, "ada__finalizationS");
   u00093 : constant Version_32 := 16#d00f339c#;
   pragma Export (C, u00093, "system__finalization_rootB");
   u00094 : constant Version_32 := 16#7a0a6580#;
   pragma Export (C, u00094, "system__finalization_rootS");
   u00095 : constant Version_32 := 16#ef3c5c6f#;
   pragma Export (C, u00095, "system__finalization_primitivesB");
   u00096 : constant Version_32 := 16#f622319e#;
   pragma Export (C, u00096, "system__finalization_primitivesS");
   u00097 : constant Version_32 := 16#9cd38c2c#;
   pragma Export (C, u00097, "system__os_locksS");
   u00098 : constant Version_32 := 16#401f6fd6#;
   pragma Export (C, u00098, "interfaces__cB");
   u00099 : constant Version_32 := 16#3dbcc8ee#;
   pragma Export (C, u00099, "interfaces__cS");
   u00100 : constant Version_32 := 16#8f29e754#;
   pragma Export (C, u00100, "system__os_constantsS");
   u00101 : constant Version_32 := 16#c04dcb27#;
   pragma Export (C, u00101, "system__os_libB");
   u00102 : constant Version_32 := 16#f51dc4c4#;
   pragma Export (C, u00102, "system__os_libS");
   u00103 : constant Version_32 := 16#94d23d25#;
   pragma Export (C, u00103, "system__atomic_operations__test_and_setB");
   u00104 : constant Version_32 := 16#57acee8e#;
   pragma Export (C, u00104, "system__atomic_operations__test_and_setS");
   u00105 : constant Version_32 := 16#b7152171#;
   pragma Export (C, u00105, "system__atomic_operationsS");
   u00106 : constant Version_32 := 16#553a519e#;
   pragma Export (C, u00106, "system__atomic_primitivesB");
   u00107 : constant Version_32 := 16#78a6d0b7#;
   pragma Export (C, u00107, "system__atomic_primitivesS");
   u00108 : constant Version_32 := 16#256dbbe5#;
   pragma Export (C, u00108, "system__stringsB");
   u00109 : constant Version_32 := 16#ebf45b4c#;
   pragma Export (C, u00109, "system__stringsS");
   u00110 : constant Version_32 := 16#fa03c63e#;
   pragma Export (C, u00110, "system__file_control_blockS");
   u00111 : constant Version_32 := 16#0ddbd91f#;
   pragma Export (C, u00111, "system__memoryB");
   u00112 : constant Version_32 := 16#68e2c74e#;
   pragma Export (C, u00112, "system__memoryS");

   --  BEGIN ELABORATION ORDER
   --  ada%s
   --  interfaces%s
   --  system%s
   --  system.atomic_operations%s
   --  system.parameters%s
   --  system.parameters%b
   --  system.crtl%s
   --  interfaces.c_streams%s
   --  interfaces.c_streams%b
   --  system.spark%s
   --  system.spark.cut_operations%s
   --  system.spark.cut_operations%b
   --  system.storage_elements%s
   --  system.img_address_32%s
   --  system.img_address_64%s
   --  system.stack_checking%s
   --  system.stack_checking%b
   --  system.string_hash%s
   --  system.string_hash%b
   --  system.htable%s
   --  system.htable%b
   --  system.strings%s
   --  system.strings%b
   --  system.traceback_entries%s
   --  system.traceback_entries%b
   --  system.unsigned_types%s
   --  system.wch_con%s
   --  system.wch_con%b
   --  system.wch_jis%s
   --  system.wch_jis%b
   --  system.wch_cnv%s
   --  system.wch_cnv%b
   --  system.traceback%s
   --  system.traceback%b
   --  system.secondary_stack%s
   --  system.standard_library%s
   --  ada.exceptions%s
   --  system.exceptions_debug%s
   --  system.exceptions_debug%b
   --  system.soft_links%s
   --  system.wch_stw%s
   --  system.wch_stw%b
   --  ada.exceptions.last_chance_handler%s
   --  ada.exceptions.last_chance_handler%b
   --  ada.exceptions.traceback%s
   --  ada.exceptions.traceback%b
   --  system.address_image%s
   --  system.address_image%b
   --  system.exception_table%s
   --  system.exception_table%b
   --  ada.numerics%s
   --  ada.numerics.big_numbers%s
   --  system.exceptions%s
   --  system.exceptions.machine%s
   --  system.exceptions.machine%b
   --  system.img_int%s
   --  system.memory%s
   --  system.memory%b
   --  system.secondary_stack%b
   --  system.soft_links.initialize%s
   --  system.soft_links.initialize%b
   --  system.soft_links%b
   --  system.standard_library%b
   --  system.traceback.symbolic%s
   --  system.traceback.symbolic%b
   --  ada.exceptions%b
   --  ada.io_exceptions%s
   --  ada.strings%s
   --  ada.strings.utf_encoding%s
   --  ada.strings.utf_encoding%b
   --  ada.strings.utf_encoding.strings%s
   --  ada.strings.utf_encoding.strings%b
   --  ada.strings.utf_encoding.wide_strings%s
   --  ada.strings.utf_encoding.wide_strings%b
   --  ada.strings.utf_encoding.wide_wide_strings%s
   --  ada.strings.utf_encoding.wide_wide_strings%b
   --  interfaces.c%s
   --  interfaces.c%b
   --  system.atomic_primitives%s
   --  system.atomic_primitives%b
   --  system.atomic_operations.test_and_set%s
   --  system.atomic_operations.test_and_set%b
   --  system.case_util%s
   --  system.case_util%b
   --  system.os_constants%s
   --  system.os_lib%s
   --  system.os_lib%b
   --  system.os_locks%s
   --  system.finalization_primitives%s
   --  system.finalization_primitives%b
   --  system.val_util%s
   --  system.val_util%b
   --  system.val_llu%s
   --  ada.tags%s
   --  ada.tags%b
   --  ada.strings.text_buffers%s
   --  ada.strings.text_buffers%b
   --  ada.strings.text_buffers.utils%s
   --  ada.strings.text_buffers.utils%b
   --  system.put_images%s
   --  system.put_images%b
   --  ada.streams%s
   --  ada.streams%b
   --  system.file_control_block%s
   --  system.finalization_root%s
   --  system.finalization_root%b
   --  ada.finalization%s
   --  system.file_io%s
   --  system.file_io%b
   --  ada.text_io%s
   --  ada.text_io%b
   --  ada_hellosekai%b
   --  END ELABORATION ORDER

end ada_main;
