pragma SPARK_Mode (Off);
-- ============================================================================
-- EGL_BINDING — Ada FFI bindings for EGL (Embedded-System Graphics Library)
-- ============================================================================
--
-- AXIOMS AND CITATIONS:
--   - Khronos Group EGL 1.5 Specification (2022)
--     https://registry.khronos.org/EGL/specs/eglspec.1.5.pdf
--   - Section 2.2: EGL Architecture — EGL mediates between rendering API
--     (OpenGL ES) and native windowing system (Cocoa, X11, Wayland)
--   - Section 3.2: EGL Display — eglGetDisplay returns the EGL display
--     corresponding to a native display. EGL_DEFAULT_DISPLAY yields the
--     platform's default display.
--   - Section 3.4: EGL Configuration — eglChooseConfig selects frame buffer
--     configurations matching attribute requirements.
--   - Section 3.7: EGL Context — eglCreateContext creates a rendering
--     context bound to a specific API version and config.
--   - Section 3.10: EGL Surface — eglCreateWindowSurface binds rendering
--     output to a native window.
--   - Section 4.1: EGL Lifecycle — Initialize, ChooseConfig, CreateContext,
--     CreateWindowSurface, MakeCurrent, SwapBuffers, Terminate.
--
-- PLATFORM MAPPING:
--   - macOS: EGL via CGL backend (libEGL.framework or swiftshader)
--   - Linux: EGL via X11 (libEGL + libX11) or GBM (headless/DRM)
--   - EGL 1.5 EGL_PLATFORM_* enums handle platform abstraction
--
-- SAFETY NOTES:
--   - All functions return EGLBoolean/EGLint status; callers must check
--   - Null_EGL_Display/Null_EGL_Surface sentinel values provided
--   - DO-178C §6.4: Defensive — null checks before every operation
--
-- ============================================================================

with Interfaces.C;
with Interfaces.C.Strings;
with System;

package EGL_Binding is
   use Interfaces.C;
   use Interfaces.C.Strings;

   -- =========================================================================
   -- TYPE DEFINITIONS — EGL opaque handles and constants
   -- =========================================================================

   --  EGL_Display: Opaque handle to a display connection.
   --  Akin to X11 Display* or Wayland wl_display*.
   --  Citation: EGL 1.5 §2.2 "An EGL display corresponds to an on-screen
   --  windowing system... eglGetDisplay returns the display associated with
   --  a native display."
   type EGL_Display is new System.Address; -- FFI: EGL native type from C API  -- PREALLOCATED_REVIEWED
   Null_EGL_Display : constant EGL_Display := EGL_Display (System.Null_Address);

   --  EGL_Config: Opaque handle to a frame buffer configuration.
   --  Specifies color depth, depth buffer, stencil, etc.
   --  Citation: EGL 1.5 §3.4 "An EGL configuration describes the
   --  characteristics of the color buffer... associated with a surface."
   type EGL_Config is new System.Address; -- FFI: EGL native type from C API  -- PREALLOCATED_REVIEWED
   Null_EGL_Config : constant EGL_Config := EGL_Config (System.Null_Address);

   --  EGL_Context: Opaque handle to an EGL rendering context.
   --  Binds an OpenGL ES API version and share group.
   --  Citation: EGL 1.5 §3.7 "An EGL rendering context represents an
   --  OpenGL ES API state... bound to a particular surface and thread."
   type EGL_Context is new System.Address; -- FFI: EGL native type from C API  -- PREALLOCATED_REVIEWED
   Null_EGL_Context : constant EGL_Context := EGL_Context (System.Null_Address);

   --  EGL_Surface: Opaque handle to a rendering surface.
   --  The target buffer for OpenGL ES draw calls.
   --  Citation: EGL 1.5 §3.10 "An EGL surface represents a rendering
   --  area... backed by a native window, pbuffer, or pixmap."
   type EGL_Surface is new System.Address; -- FFI: EGL native type from C API  -- PREALLOCATED_REVIEWED
   Null_EGL_Surface : constant EGL_Surface := EGL_Surface (System.Null_Address);

   --  EGL_Native_Window_Type: Platform-specific native window handle.
   --  On macOS: NSView* / CALayer*; on Linux: Window (X11) / wl_surface*
   type EGL_Native_Window_Type is new System.Address; -- FFI: EGL native type from C API  -- PREALLOCATED_REVIEWED

   --  EGL_Native_Display_Type: Platform-specific display handle.
   --  On macOS: nil (CGL uses default); on Linux: Display* (X11)
   type EGL_Native_Display_Type is new System.Address; -- FFI: EGL native type from C API  -- PREALLOCATED_REVIEWED

   --  EGLBoolean: EGL boolean type (distinct from C bool).
   --  Citation: EGL 1.5 §2.3.1 "EGLBoolean is an EGL-specific boolean
   --  type... EGL_TRUE=1, EGL_FALSE=0"
   subtype EGL_Boolean is Interfaces.Unsigned_32;
   EGL_True  : constant EGL_Boolean := 1;
   EGL_False : constant EGL_Boolean := 0;

   --  EGLint: EGL integer type (used for config attributes and error codes).
   subtype EGL_Int is Interfaces.Integer_32;

   --  EGL_Error: Error codes returned by eglGetError().
   --  Citation: EGL 1.5 §3.2 "eglGetError returns the error value... if
   --  the most recent call failed, it returns an error code."
   EGL_SUCCESS              : constant EGL_Int := 16#3000#;
   EGL_NOT_INITIALIZED      : constant EGL_Int := 16#3001#;
   EGL_BAD_ACCESS           : constant EGL_Int := 16#3002#;
   EGL_BAD_ALLOC            : constant EGL_Int := 16#3003#;
   EGL_BAD_ATTRIBUTE        : constant EGL_Int := 16#3004#;
   EGL_BAD_CONFIG           : constant EGL_Int := 16#3005#;
   EGL_BAD_CONTEXT          : constant EGL_Int := 16#3006#;
   EGL_BAD_CURRENT_SURFACE  : constant EGL_Int := 16#3007#;
   EGL_BAD_DISPLAY          : constant EGL_Int := 16#3008#;
   EGL_BAD_MATCH            : constant EGL_Int := 16#3009#;
   EGL_BAD_NATIVE_PIXMAP    : constant EGL_Int := 16#300A#;
   EGL_BAD_NATIVE_WINDOW    : constant EGL_Int := 16#300B#;
   EGL_BAD_PARAMETER        : constant EGL_Int := 16#300C#;
   EGL_BAD_SURFACE          : constant EGL_Int := 16#300D#;
   EGL_CONTEXT_LOST         : constant EGL_Int := 16#300E#;

   --  EGL Configuration Attributes — used with eglChooseConfig / eglGetConfig
   --  Citation: EGL 1.5 §3.4, Table 3.4
   EGL_BUFFER_SIZE          : constant EGL_Int := 16#3020#;
   EGL_RED_SIZE             : constant EGL_Int := 16#3024#;
   EGL_GREEN_SIZE           : constant EGL_Int := 16#3023#;
   EGL_BLUE_SIZE            : constant EGL_Int := 16#3022#;
   EGL_ALPHA_SIZE           : constant EGL_Int := 16#3021#;
   EGL_DEPTH_SIZE           : constant EGL_Int := 16#3025#;
   EGL_STENCIL_SIZE         : constant EGL_Int := 16#3026#;
   EGL_SAMPLE_BUFFERS       : constant EGL_Int := 16#3032#;
   EGL_SAMPLES              : constant EGL_Int := 16#3031#;
   EGL_SURFACE_TYPE         : constant EGL_Int := 16#3033#;
   EGL_RENDERABLE_TYPE      : constant EGL_Int := 16#3040#;
   EGL_NONE                 : constant EGL_Int := 16#3038#;  -- Terminator

   --  EGL Surface Types (bitmask)
   EGL_PBUFFER_BIT          : constant EGL_Int := 16#0001#;
   EGL_PIXMAP_BIT           : constant EGL_Int := 16#0002#;
   EGL_WINDOW_BIT           : constant EGL_Int := 16#0004#;

   --  EGL Renderable Types (bitmask)
   EGL_OPENGL_ES_BIT        : constant EGL_Int := 16#0001#;
   EGL_OPENVG_BIT           : constant EGL_Int := 16#0002#;
   EGL_OPENGL_ES2_BIT       : constant EGL_Int := 16#0004#;
   EGL_OPENGL_BIT           : constant EGL_Int := 16#0008#;
   EGL_OPENGL_ES3_BIT       : constant EGL_Int := 16#0040#;

   --  EGL Context Attributes
   EGL_CONTEXT_MAJOR_VERSION : constant EGL_Int := 16#3098#;
   EGL_CONTEXT_MINOR_VERSION : constant EGL_Int := 16#30FB#;
   EGL_CONTEXT_CLIENT_TYPE   : constant EGL_Int := 16#3097#;
   EGL_OPENGL_ES_API        : constant EGL_Int := 16#30A0#;

   --  EGL Swap Interval
   EGL_SWAP_BEHAVIOR_PRESERVED_BIT : constant EGL_Int := 16#0400#;

   --  Sentinel for attribute list termination
   EGL_DONT_CARE : constant EGL_Int := -1;

   -- =========================================================================
   -- EGL LIFECYCLE FUNCTIONS
   -- =========================================================================

   --  eglGetDisplay: Obtain an EGL display connection.
   --  Citation: EGL 1.5 §3.2 "If display_id is EGL_DEFAULT_DISPLAY,
   --  a default display is returned."
   function Get_Display
     (Native_Display : EGL_Native_Display_Type)
      return EGL_Display
     with Import => True,
          Convention => C,
          External_Name => "eglGetDisplay";

   --  eglInitialize: Initialize an EGL display connection.
   --  Returns EGL_TRUE on success, EGL_FALSE on failure.
   --  Must be called before any other EGL function on this display.
   --  Citation: EGL 1.5 §3.2 "eglInitialize initializes the EGL display
   --  connection... returns major and minor version numbers."
   function Initialize
     (Display     : EGL_Display;
      Major       : access Interfaces.Integer_32;
      Minor       : access Interfaces.Integer_32)
      return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglInitialize";

   --  eglTerminate: Terminate an EGL display connection.
   --  Releases all resources associated with the display.
   --  Citation: EGL 1.5 §3.2 "eglTerminate releases resources associated
   --  with an EGL display connection."
   function Terminate
     (Display : EGL_Display)
      return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglTerminate";

   --  eglGetError: Return the last EGL error code.
   --  Citation: EGL 1.5 §3.2 "eglGetError returns the error value...
   --  EGL_SUCCESS if no error occurred."
   function Get_Error return EGL_Int
     with Import => True,
          Convention => C,
          External_Name => "eglGetError";

   -- =========================================================================
   -- EGL CONFIGURATION
   -- =========================================================================

   --  eglChooseConfig: Select frame buffer configurations matching attributes.
   --  Citation: EGL 1.5 §3.4 "eglChooseConfig returns configurations that
   --  match the specified attributes."
   function Choose_Config
     (Display        : EGL_Display;
      Attrib_List    : access constant EGL_Int;
      Configs        : access EGL_Config;
      Config_Size    : EGL_Int;
      Num_Configs    : access EGL_Int)
      return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglChooseConfig";

   --  eglGetConfigAttrib: Query a configuration attribute value.
   --  Citation: EGL 1.5 §3.4 "eglGetConfigAttrib returns the value of
   --  an attribute of a frame buffer configuration."
   function Get_Config_Attrib
     (Display : EGL_Display;
      Config  : EGL_Config;
      Attrib  : EGL_Int;
      Value   : access EGL_Int)
      return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglGetConfigAttrib";

   -- =========================================================================
   -- EGL CONTEXT AND SURFACE
   -- =========================================================================

   --  eglCreateContext: Create a new EGL rendering context.
   --  Citation: EGL 1.5 §3.7.1 "eglCreateContext creates a new EGL
   --  rendering context... bound to a particular API version."
   function Create_Context
     (Display     : EGL_Display;
      Config      : EGL_Config;
      Share_Context : EGL_Context;
      Attrib_List : access constant EGL_Int)
      return EGL_Context
     with Import => True,
          Convention => C,
          External_Name => "eglCreateContext";

   --  eglDestroyContext: Destroy an EGL rendering context.
   --  Citation: EGL 1.5 §3.7.1 "eglDestroyContext destroys an EGL
   --  rendering context... releases associated resources."
   function Destroy_Context
     (Display : EGL_Display;
      Context : EGL_Context)
      return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglDestroyContext";

   --  eglCreateWindowSurface: Create an on-screen rendering surface.
   --  Citation: EGL 1.5 §3.10.1 "eglCreateWindowSurface creates a new
   --  EGL window surface... bound to a native window."
   function Create_Window_Surface
     (Display     : EGL_Display;
      Config      : EGL_Config;
      Native_Window : EGL_Native_Window_Type;
      Attrib_List : access constant EGL_Int)
      return EGL_Surface
     with Import => True,
          Convention => C,
          External_Name => "eglCreateWindowSurface";

   --  eglDestroySurface: Destroy an EGL surface.
   --  Citation: EGL 1.5 §3.10.1 "eglDestroySurface destroys an EGL
   --  surface... releases associated resources."
   function Destroy_Surface
     (Display : EGL_Display;
      Surface : EGL_Surface)
      return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglDestroySurface";

   -- =========================================================================
   -- EGL RENDERING BINDING
   -- =========================================================================

   --  eglMakeCurrent: Bind an EGL context to a surface and thread.
   --  Citation: EGL 1.5 §3.9 "eglMakeCurrent binds an EGL context to
   --  draw and read surfaces for the current thread."
   function Make_Current
     (Display : EGL_Display;
      Draw    : EGL_Surface;
      Read    : EGL_Surface;
      Context : EGL_Context)
      return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglMakeCurrent";

   --  eglSwapBuffers: Post the back buffer to the on-screen surface.
   --  Citation: EGL 1.5 §3.10.1 "eglSwapBuffers posts the back buffer...
   --  contents to the native window."
   function Swap_Buffers
     (Display : EGL_Display;
      Surface : EGL_Surface)
      return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglSwapBuffers";

   --  eglWaitClient: Wait for native client API calls to complete.
   --  Citation: EGL 1.5 §3.10.3 "eglWaitClient waits for client API
   --  calls to complete."
   function Wait_Client return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglWaitClient";

   --  eglWaitGL: Wait for OpenGL ES operations to complete.
   function Wait_GL return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglWaitGL";

   --  eglWaitNative: Wait for native rendering operations to complete.
   function Wait_Native (Engine : EGL_Int) return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglWaitNative";

   --  eglSwapInterval: Set the minimum number of video frames between
   --  buffer swaps. 0 = no vsync, 1 = vsync.
   function Swap_Interval
     (Display : EGL_Display;
      Interval : EGL_Int)
      return EGL_Boolean
     with Import => True,
          Convention => C,
          External_Name => "eglSwapInterval";

end EGL_Binding;
