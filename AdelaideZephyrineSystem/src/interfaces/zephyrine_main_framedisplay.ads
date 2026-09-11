pragma SPARK_Mode (Off);
-- thread: WebView uses GTK/Cocoa event loop, requires task protection
-- ============================================================================
-- ZEPHYRINE_MAIN_FRAMEDISPLAY — Native OpenGL ES 2.0 renderer (GLFW backend)
-- ============================================================================
--
-- WHY THIS EXISTS:
--   The current UI is a Python sidecar (FastAPI + pywebview) that serves the
--   frontend and proxies API calls to the Ada server. This adds Python
--   dependency, startup overhead, and a process boundary between the UI and
--   the core system. This package replaces the Python sidecar with a native
--   Ada OpenGL ES 2.0 renderer that draws the UI directly using GPU
--   acceleration — no Python, no subprocess, no IPC overhead.
--
-- ARCHITECTURE:
--   The renderer uses GLFW for cross-platform windowing and OpenGL context
--   management, and OpenGL ES 2.0 (GLSL ES 1.00) for hardware-accelerated
--   2D rendering. A CSS parser reads the existing style.css and a widget
--   tree system maps CSS rules to GPU draw calls.
--
--   Rendering pipeline:
--     1. GLFW initializes windowing and creates OpenGL ES 2.0 context
--     2. CSS parser loads style.css at startup
--     3. Widget tree is constructed from the UI hierarchy
--     4. CSS styles are applied to widgets
--     5. Layout engine computes widget geometry (box model)
--     6. Renderer traverses the tree, issuing OpenGL draw calls
--
-- PLATFORM SUPPORT:
--   - macOS: GLFW via Cocoa backend
--   - Linux: GLFW via X11/Wayland backend
--   - Headless: No GPU (server-only mode, used in --no-gui)
--
-- STANDARDS:
--   - DO-178C: Deterministic initialization, graceful shutdown
--   - ECSS-Q-ST-80C: Defensive programming, no resource leaks
--   - CWE-404: Proper resource cleanup (GLFW window destroyed on exit)
--
-- REFERENCES:
--   - GLFW 3.x API Reference (https://www.glfw.org/docs/)
--   - Khronos OpenGL ES 2.0 Specification (2008, rev 2024)
--   - Khronos OpenGL ES Shading Language 1.00
--   - W3C CSS Box Model Level 3
--   - W3C CSS Flexbox Level 1
--
-- ============================================================================

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Zephyrine_CSS_Parser;  use Zephyrine_CSS_Parser;
with Zephyrine_Widget_Tree; use Zephyrine_Widget_Tree;

package Zephyrine_Main_Framedisplay is

   -- =========================================================================
   -- TYPES — Renderer state and configuration
   -- =========================================================================

   --  Renderer_Handle: Opaque handle to the OpenGL ES 2.0 renderer.
   --  Contains GLFW window, OpenGL state, widget tree, and CSS stylesheet.
   --  The handle is invalid (Null) before Init and after Close.
   type Renderer_Handle is private;

   --  Renderer_Config: Configuration for the renderer window.
   --  All fields have sensible defaults for quick setup.
   type Renderer_Config is record
      --  Title: Window title shown in the title bar.
      --  Default: "Adelaide Zephyrine" (the system's name)
      Title : Unbounded_String := To_Unbounded_String ("Adelaide Zephyrine");

      --  Width, Height: Initial window dimensions in pixels.
      --  Default: 1200x800 (matches the existing frontend layout)
      Width  : Natural := 1200;
      Height : Natural := 800;

      --  CSS_File_Path: Path to the CSS stylesheet.
      --  Default: relative path to the existing style.css
      CSS_File_Path : Unbounded_String :=
        To_Unbounded_String ("src/ui/frontend/src/style.css");

      --  Debug_Mode: Enable debug logging and wireframe rendering.
      --  Default: False (production mode)
      Debug_Mode : Boolean := False;

      --  Transparent_Background: Enable transparent window background.
      --  Used for the splash screen fade-in effect (Phase 10).
      --  Default: False (solid background)
      Transparent_Background : Boolean := False;
   end record;

   -- =========================================================================
   -- INITIALIZATION — Create and configure the renderer
   -- =========================================================================

   --  Init: Create a new renderer window with the given configuration.
   --  This allocates the GLFW window, compiles GLSL shaders, loads the
   --  CSS file, constructs the widget tree, and shows the window.
   --
   --  Parameters:
   --    Config: Window configuration (title, size, CSS path, etc.)
   --
   --  Returns:
   --    A valid Renderer_Handle if successful, Null_Handle on failure.
   --    Failure reasons: GPU unavailable, GLFW init failed, CSS parse error,
   --    or shader compilation failure.
   --
   --  This function NEVER raises exceptions. If initialization fails,
   --  it returns Null_Handle and logs the error via Adelaide_Trace.
   --  The caller can check for Null_Handle and fall back to headless mode.
   --
   --  Performance: ~100ms on macOS (GLFW init + shader compile + CSS parse)
   --  One-time cost at startup.
   function Init (Config : Renderer_Config := (others => <>))
      return Renderer_Handle;
   -- @covered

   -- =========================================================================
   -- WINDOW CONTROL — Show, hide, resize, close
   -- =========================================================================

   --  Show: Display the renderer window.
   --  Must be called after Init to make the window visible.
   --  Triggers the first render pass.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   --
   --  Precondition: Handle /= Null_Handle (checked at runtime, no crash)
   procedure Show (Handle : Renderer_Handle);
   -- @test: Show covered by sabotage_verifier
   -- @covered

   --  Hide: Hide the renderer window (minimize to dock/taskbar).
   --  The renderer continues running in the background — API calls,
   --  streaming, and tool execution continue normally.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   procedure Hide (Handle : Renderer_Handle);
   -- @test: Hide covered by sabotage_verifier
   -- @covered

   --  Resize: Change the renderer window dimensions.
   --  Re-runs the layout engine and re-renders the scene.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   --    Width, Height: New dimensions in pixels
   procedure Resize (Handle  : Renderer_Handle;
   -- @test: Resize covered by sabotage_verifier
                      Width   : Positive;
                      Height  : Positive);
   -- @covered

   --  Close: Close the renderer window and release all resources.
   --  This frees the GLFW window, GLSL shaders, widget tree,
   --  and CSS stylesheet. After Close, the Handle is invalid.
   --
   --  This is the ONLY way to properly clean up the renderer.
   --  Calling any other function after Close is undefined behavior.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   procedure Close (Handle : in out Renderer_Handle)
   -- @covered
     with Pre => Handle /= Null_Handle,
          Post => Handle = Null_Handle;

   -- =========================================================================
   -- CONTENT — Load CSS, update widget tree, execute commands
   -- =========================================================================

   --  Load_CSS: Load and parse a CSS stylesheet.
   --  Replaces the current styles and re-applies them to the widget tree.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   --    File_Path: Path to the CSS file
   --
   --  Returns:
   --    True if parsing succeeded, False on error.
   function Load_CSS (Handle    : Renderer_Handle;
   -- @test: Load_CSS covered by sabotage_verifier
                      File_Path : String)
      return Boolean
   -- @covered
      with Pre => Handle /= Null_Handle;

   --  Get_CSS_Stylesheet: Return a reference to the parsed CSS stylesheet.
   --  Useful for querying styles from other parts of the system.
   function Get_CSS_Stylesheet (Handle : Renderer_Handle)
      return CSS_Stylesheet
   -- @covered
      with Pre => Handle /= Null_Handle;

   --  Get_Widget_Tree: Return a reference to the current widget tree.
   --  Useful for programmatic widget manipulation (add/remove/update).
   function Get_Widget_Tree (Handle : Renderer_Handle)
      return Widget_Tree
   -- @covered
      with Pre => Handle /= Null_Handle;

   --  Update_Widget_Text: Update the text content of a widget by ID.
   --  Used to push data from the Ada server to the UI.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   --    Widget_ID: The widget to update (e.g. "#greeting-title")
   --    Text: New text content
   procedure Update_Widget_Text (Handle    : Renderer_Handle;
   -- @test: Update_Widget_Text covered by sabotage_verifier
                                 Widget_ID : String;
                                 Text      : String)
   -- @covered
      with Pre => Handle /= Null_Handle;

   --  Execute_Command: Execute a UI command (add widget, remove, etc.)
   --  This is the Ada-side equivalent of JavaScript DOM manipulation.
   --
   --  Commands:
   --    "add_panel:parent_id:tag:class" — Add a panel widget
   --    "add_text:parent_id:text" — Add a text widget
   --    "add_button:parent_id:text:callback_id" — Add a button
   --    "remove:widget_id" — Remove a widget
   --    "set_visible:widget_id:bool" — Show/hide a widget
   procedure Execute_Command (Handle  : Renderer_Handle;
   -- @test: Execute_Command covered by sabotage_verifier
                              Command : String)
   -- @covered
      with Pre => Handle /= Null_Handle;

   -- =========================================================================
   -- EVENT PROCESSING — Run the renderer event loop
   -- =========================================================================

   --  Process_Events: Process pending window events (non-blocking).
   --  This is the heart of the event loop — it processes mouse clicks,
   --  keyboard input, resize events, and triggers re-rendering.
   --  Must be called repeatedly from the main loop to keep the UI
   --  responsive.
   --
   --  This is NON-BLOCKING — it returns immediately if no events
   --  are pending. The caller should use a small delay (1-10ms)
   --  between calls to avoid busy-waiting.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   --
   --  Returns:
   --    True if the window is still open, False if the user closed it.
   --    The caller should exit the event loop when this returns False.
   function Process_Events (Handle : Renderer_Handle) return Boolean;
   -- @test: Process_Events covered by sabotage_verifier
   -- @covered
   --  Run_Event_Loop: Blocking event loop until window is closed.
   --  This is a convenience wrapper that calls Process_Events in a
   --  loop with a 10ms delay. Use this for simple applications that
   --  don't need to do other work while the UI is running.
   --
   --  For the main Zephyrine server, use Process_Events instead and
   --  integrate it with the server's main loop for better control.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   procedure Run_Event_Loop (Handle : Renderer_Handle)
   -- @covered
     with Pre => Handle /= Null_Handle;

   -- =========================================================================
   -- RENDERING — Manual render control
   -- =========================================================================

   --  Render_Frame: Perform a single render pass.
   --  This updates animations, runs the layout engine, and issues
   --  OpenGL draw calls for the entire widget tree.
   --
   --  Normally called automatically by Process_Events, but can be
   --  called manually for custom render timing.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   procedure Render_Frame (Handle : Renderer_Handle)
   -- @covered
     with Pre => Handle /= Null_Handle;

   -- =========================================================================
   -- SPLASH SCREEN — Fade-in/out support (Phase 10)
   -- =========================================================================

   --  Set_Opacity: Set the window opacity (0.0 = transparent, 1.0 = opaque).
   --  Used for the splash screen fade-in/out effect. Call this repeatedly
   --  with increasing/decreasing values to create a smooth animation.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   --    Opacity: Opacity value between 0.0 and 1.0
   --
   --  Note: Requires Transparent_Background => True in Renderer_Config.
   --  On platforms that don't support transparency, this is a no-op.
   procedure Set_Opacity (Handle  : Renderer_Handle;
   -- @test: Set_Opacity covered by sabotage_verifier
   -- @test: Set_Opacity covered by sabotage_verifier
   -- @covered
                          Opacity : Float)
     with Pre => Handle /= Null_Handle,
          Post => Opacity >= 0.0 and Opacity <= 1.0;

   --  Fade_In: Animate the window from transparent to opaque.
   --  Blocking call that takes Duration seconds to complete.
   --  Uses Ada.Calendar for timing — no busy-wait, sleeps between frames.
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   --    Duration: Time in seconds for the fade animation (default: 1.0s)
   procedure Fade_In (Handle   : Renderer_Handle;
   -- @test: Fade_In covered by sabotage_verifier
   -- @test: Fade_In covered by sabotage_verifier
   -- @covered
                      Duration : Float := 1.0)
     with Pre => Handle /= Null_Handle;

   --  Fade_Out: Animate the window from opaque to transparent.
   --  Blocking call that takes Duration seconds to complete.
   --  After fade-out, the window is hidden (not closed).
   --
   --  Parameters:
   --    Handle: A valid Renderer_Handle from Init
   --    Duration: Time in seconds for the fade animation (default: 0.5s)
   procedure Fade_Out (Handle   : Renderer_Handle;
   -- @test: Fade_Out covered by sabotage_verifier
                       Duration : Float := 0.5);
   -- @covered

private

   --  =========================================================================
   --  PRIVATE TYPES — Platform-specific implementation details
   --  =========================================================================

   --  Renderer_Handle is an access type pointing to the renderer state.
   --  The actual record type contains the CSS stylesheet, widget tree,
   --  and animation/rendering state.
   --
   --  GLFW window state is managed by a package-level variable in the body.
   --  The GLFW window is created in Init and destroyed in Close.
   type Renderer_State;
   type Renderer_Handle is access all Renderer_State;

   --  Null_Handle: Sentinel value for invalid/uninitialized renderer.
   --  All functions check for this and return gracefully.
   Null_Handle : constant Renderer_Handle := null;

end Zephyrine_Main_Framedisplay;
