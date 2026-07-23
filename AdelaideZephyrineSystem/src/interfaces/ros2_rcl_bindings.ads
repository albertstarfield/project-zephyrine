with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

package ROS2_RCL_Bindings is
   --  Thin bindings to ROS2 RCL C API
   
   type rcl_context_t is record
      global_arguments : System.Address; -- FFI: System.Address required for C binding
      impl : System.Address; -- FFI: System.Address required for C binding
      instance_id_storage : Interfaces.C.size_t;
   end record;
   pragma Convention (C, rcl_context_t);

   type rcl_node_t is record
      context : System.Address; -- FFI: System.Address required for C binding
      impl : System.Address; -- FFI: System.Address required for C binding
   end record;
   pragma Convention (C, rcl_node_t);
   
   type rcl_publisher_t is record
      impl : System.Address; -- FFI: System.Address required for C binding
   end record;
   pragma Convention (C, rcl_publisher_t);

   type rcl_subscription_t is record
      impl : System.Address; -- FFI: System.Address required for C binding
   end record;
   pragma Convention (C, rcl_subscription_t);

   type rcl_allocator_t is record
      allocate : System.Address; -- FFI: System.Address required for C binding
      deallocate : System.Address; -- FFI: System.Address required for C binding
      reallocate : System.Address; -- FFI: System.Address required for C binding
      zero_allocate : System.Address; -- FFI: System.Address required for C binding
      state : System.Address; -- FFI: System.Address required for C binding
   end record;
   pragma Convention (C, rcl_allocator_t);
   
   type rcl_init_options_t is record
      impl : System.Address; -- FFI: System.Address required for C binding
   end record;
   pragma Convention (C, rcl_init_options_t);

   type rcl_node_options_t is record
      allocator : rcl_allocator_t;
      use_global_arguments : Interfaces.C.char;
      arguments : System.Address; -- FFI: System.Address required for C binding
      enable_rosout : Interfaces.C.char;
      rosout_qos : System.Address; -- FFI: System.Address required for C binding
   end record;
   pragma Convention (C, rcl_node_options_t);

   type rcl_ret_t is new Interfaces.C.int;
   RCL_RET_OK : constant rcl_ret_t := 0;

    --  Returns the system default memory allocator for ROS2.
    function rcutils_get_default_allocator return rcl_allocator_t;
   pragma Import (C, rcutils_get_default_allocator, "rcutils_get_default_allocator");

    --  Returns a zero-initialized RCL init options struct.
    function rcl_get_zero_initialized_init_options return rcl_init_options_t;
   pragma Import (C, rcl_get_zero_initialized_init_options, "rcl_get_zero_initialized_init_options");

    --  Initializes the RCL init options struct with the given allocator.
    --  Initializes the RCL context with command-line arguments and init options.
    function rcl_init_options_init
     (options : access rcl_init_options_t;
      allocator : rcl_allocator_t) return rcl_ret_t;
   pragma Import (C, rcl_init_options_init, "rcl_init_options_init");

    --  Returns a zero-initialized RCL context struct.
    function rcl_get_zero_initialized_context return rcl_context_t;
   pragma Import (C, rcl_get_zero_initialized_context, "rcl_get_zero_initialized_context");

   --  rcl_init: C FFI binding to initialize the ROS2 client library.
   function rcl_init
     (argc : Interfaces.C.int;
      argv : System.Address; -- FFI: System.Address required for C binding
      options : access rcl_init_options_t;
      context : access rcl_context_t) return rcl_ret_t;
   pragma Import (C, rcl_init, "rcl_init");

    --  Returns a zero-initialized RCL node struct.
    function rcl_get_zero_initialized_node return rcl_node_t;
   pragma Import (C, rcl_get_zero_initialized_node, "rcl_get_zero_initialized_node");

    --  Returns the default node options with standard configuration.
    function rcl_node_get_default_options return rcl_node_options_t;
   pragma Import (C, rcl_node_get_default_options, "rcl_node_get_default_options");

    --  Initializes an RCL node with the given name, namespace, context, and options.
    function rcl_node_init
     (node : access rcl_node_t;
      name : chars_ptr;
      namespace : chars_ptr;
      context : access rcl_context_t;
      options : access rcl_node_options_t) return rcl_ret_t;
   pragma Import (C, rcl_node_init, "rcl_node_init");
   
end ROS2_RCL_Bindings;
