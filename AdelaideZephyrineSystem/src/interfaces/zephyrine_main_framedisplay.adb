pragma SPARK_Mode (Off);
-- thread: WebView uses GTK/Cocoa event loop, requires task protection
-- ============================================================================
-- ZEPHYRINE_MAIN_FRAMEDISPLAY — Native OpenGL ES 2.0 renderer (GLFW backend)
-- ============================================================================
--
-- AXIOMS AND CITATIONS:
--   - GLFW 3.x API: Window creation, context management, event polling
--   - OpenGL ES 2.0 §3.5-3.6: Shader compilation and program linking
--   - OpenGL ES 2.0 §3.8: Drawing primitives (glDrawArrays)
--   - W3C CSS Box Model Level 3: Content -> Padding -> Border -> Margin
--   - CSS Flexbox Level 1 §5.1: Main axis layout algorithm
--
-- IMPLEMENTATION NOTES:
--   - Uses GLFW for cross-platform windowing and OpenGL context management
--   - OpenGL ES 2.0 (GLSL ES 1.00) for 2D rendering: solid-color quads with alpha
--   - CSS parser reads the existing style.css at startup
--   - Widget tree maps HTML elements to GPU draw calls
--   - Layout engine computes box model geometry (simplified flexbox)
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

with Ada.Text_IO;            use Ada.Text_IO;
with Ada.Calendar;           use Ada.Calendar;
with Ada.Real_Time;          use Ada.Real_Time;
with Interfaces.C;
with Glfw;
with Glfw.Windows;
with Glfw.Windows.Context;
with Glfw.Windows.Hints;
with Glfw.Input;
with GL.Window;
with GL.Types;
with Zephyrine_CSS_Parser;   use Zephyrine_CSS_Parser;
with Zephyrine_Widget_Tree;  use Zephyrine_Widget_Tree;
with Adelaide_Trace;

package body Zephyrine_Main_Framedisplay is

   -- =========================================================================
   -- PACKAGE-LEVEL STATE — GLFW window (single-window application)
   -- =========================================================================

   --  Main_Window: The GLFW window handle.
   --  Stored as a package-level variable because GLFW Window is a controlled
   --  type (derives from Ada.Finalization.Controlled) and cannot be stored
   --  in a plain record. The window is created in Init and destroyed in Close.
   Main_Window : Glfw.Windows.Window;

   -- =========================================================================
   -- PRIVATE TYPES — Renderer state (defined in body for encapsulation)
   -- =========================================================================

   --  Renderer_State: Complete renderer state.
   --  Contains CSS stylesheet, widget tree, and animation/rendering state.
   --  GLFW window state is managed by the package-level Main_Window variable.
   type Renderer_State is record
      --  Configuration
      Config        : Renderer_Config;

      --  Parsed CSS stylesheet
      Stylesheet    : CSS_Stylesheet;

      --  Widget tree
      Tree          : Widget_Tree;

      --  Rendering state
      Is_Visible    : Boolean := False;
      Is_Initialized : Boolean := False;
      Current_Opacity : Float := 1.0;

      --  Animation timing
      Last_Frame_Time : Ada.Real_Time.Time := Ada.Real_Time.Clock;
      Delta_Time      : Float := 0.0;
   end record;

   -- =========================================================================
   -- INITIALIZATION
   -- =========================================================================

   function Init (Config : Renderer_Config := (others => <>))
      return Renderer_Handle
   is
      Handle : Renderer_Handle;
   begin
      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:init",
        Message => "Initializing OpenGL ES 2.0 renderer (GLFW backend): " &
          To_String (Config.Title) & " " &
          Natural'Image (Config.Width) & "x" &
          Natural'Image (Config.Height));

      -- Allocate renderer state
      Handle := new Renderer_State;
      Handle.Config := Config;

      --  Initialize GLFW
      --  Citation: GLFW 3.x "glfwInit initializes the GLFW library.
      --  Before library can be used, this function must be called."
      --  Note: GLFW.Init also calls GL.Init per OpenGLAda convention.
      begin
         Glfw.Init;
      exception
         when others =>
            Adelaide_Trace.Trace_Print (
              Toolcall => "framedisplay:init",
              Message => "ERROR: Glfw.Init failed — no GPU or windowing system available");
            Free (Handle);
            return Null_Handle;
      end;

      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:init",
        Message => "GLFW initialized");

      --  Set window hints for OpenGL ES 2.0
      --  Citation: GLFW 3.x "Window hints must be set before window creation.
      --  They control the framebuffer format, API version, and behavior."
      Glfw.Windows.Hints.Reset_To_Defaults;
      Glfw.Windows.Hints.Set_Client_API (Glfw.Windows.Context.OpenGL_ES);
      Glfw.Windows.Hints.Set_Minimum_OpenGL_Version (2, 0);
      Glfw.Windows.Hints.Set_Color_Bits (8, 8, 8, 8);   -- RGBA
      Glfw.Windows.Hints.Set_Depth_Bits (24);             -- Depth buffer
      Glfw.Windows.Hints.Set_Stencil_Bits (8);            -- Stencil buffer
      Glfw.Windows.Hints.Set_Doublebuffer (True);         -- Double buffering
      Glfw.Windows.Hints.Set_Resizable (True);            -- Allow resize

      if Config.Transparent_Background then
         Glfw.Windows.Hints.Set_Transparent_Framebuffer (True);
      end if;

      --  Create the GLFW window
      --  Citation: GLFW 3.x "glfwCreateWindow creates a window with the
      --  given parameters and its associated OpenGL context."
      begin
         Main_Window.Init (
            Width  => Glfw.Size (Config.Width),
            Height => Glfw.Size (Config.Height),
            Title  => To_String (Config.Title));
      exception
         when others =>
            Adelaide_Trace.Trace_Print (
              Toolcall => "framedisplay:init",
              Message => "ERROR: Glfw.Windows.Init failed — window creation failed");
            Glfw.Shutdown;
            Free (Handle);
            return Null_Handle;
      end;

      --  Make the OpenGL context current
      --  Citation: GLFW 3.x "glfwMakeContextCurrent makes the OpenGL context
      --  of the specified window current on the calling thread."
      Glfw.Windows.Context.Make_Current (Main_Window'Access);
      Main_Window.Show;

      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:init",
        Message => "GLFW window created and context made current");

      --  Load CSS stylesheet
      declare
         CSS_Path : constant String := To_String (Config.CSS_File_Path);
         Parse_Ok : Boolean;
      begin
         Parse_Ok := Parse_CSS_File (CSS_Path, Handle.Stylesheet);
         if not Parse_Ok then
            Adelaide_Trace.Trace_Print (
              Toolcall => "framedisplay:init",
              Message => "WARNING: CSS parse failed, using defaults");
            -- Continue with default styles (not fatal)
         else
            Adelaide_Trace.Trace_Print (
              Toolcall => "framedisplay:init",
              Message => "CSS loaded: " &
                Natural'Image (Handle.Stylesheet.Rule_Count) & " rules");
         end if;
      end;

      --  Construct initial widget tree for the Zephyrine UI
      --  This maps the HTML structure from the existing frontend:
      --    #app -> #sidebar + #main-area
      --    #sidebar: .sidebar-top + .sidebar-bottom
      --    #main-area: #empty-state | #chat-container
      declare
         Root_ID     : Widget_ID;
         Sidebar_ID  : Widget_ID;
         Main_Area_ID : Widget_ID;
         Sidebar_Top : Widget_ID;
         Sidebar_Bot : Widget_ID;
         Nav_Chat    : Widget_ID;
         Nav_About   : Widget_ID;
         Nav_Knowledge : Widget_ID;
         Nav_Agentic : Widget_ID;
         New_Chat_Btn : Widget_ID;
         Greeting_ID : Widget_ID;
      begin
         Init_Handle : declare
         begin
            Init_Tree (Handle.Tree);
         end Init_Handle;

         -- Build the UI hierarchy
         --  #app (root container, flex row)
         Root_ID := Add_Widget (Handle.Tree, Widget_Container, "app", "app", 1);
         Handle.Tree.Widgets (Root_ID).Layout.Direction := Dir_Row;

         --  #sidebar (fixed width 260px, glass effect)
         Sidebar_ID := Add_Widget (Handle.Tree, Widget_Panel, "sidebar", "sidebar", Root_ID);
         Handle.Tree.Widgets (Sidebar_ID).Layout.Direction := Dir_Column;
         Handle.Tree.Widgets (Sidebar_ID).Layout.Margin := (16.0, 16.0, 16.0, 16.0);

         --  .sidebar-top (flex column, contains nav items)
         Sidebar_Top := Add_Widget (Handle.Tree, Widget_Container, "sidebar-top", "sidebar-top", Sidebar_ID);
         Handle.Tree.Widgets (Sidebar_Top).Layout.Direction := Dir_Column;
         Handle.Tree.Widgets (Sidebar_Top).Layout.Flex_Grow := 1.0;

         --  .new-chat-btn (pill-shaped button)
         New_Chat_Btn := Add_Widget (Handle.Tree, Widget_Button, "new-chat-btn", "new-chat-btn", Sidebar_Top);
         Handle.Tree.Widgets (New_Chat_Btn).Text_Content := To_Unbounded_String ("+ New Chat");
         Handle.Tree.Widgets (New_Chat_Btn).Layout.Direction := Dir_Row;
         Handle.Tree.Widgets (New_Chat_Btn).Layout.Border_Radius := 9999.0;  -- Pill shape
         Handle.Tree.Widgets (New_Chat_Btn).Layout.Padding := (12.0, 16.0, 12.0, 16.0);

         --  .sidebar-bottom (navigation items)
         Sidebar_Bot := Add_Widget (Handle.Tree, Widget_Container, "sidebar-bottom", "sidebar-bottom", Sidebar_ID);
         Handle.Tree.Widgets (Sidebar_Bot).Layout.Direction := Dir_Column;

         --  Navigation items
         Nav_Chat := Add_Widget (Handle.Tree, Widget_Button, "nav-chat", "nav-item", Sidebar_Bot);
         Handle.Tree.Widgets (Nav_Chat).Text_Content := To_Unbounded_String ("Chat");
         Handle.Tree.Widgets (Nav_Chat).Layout.Direction := Dir_Row;

         Nav_About := Add_Widget (Handle.Tree, Widget_Button, "nav-about", "nav-item", Sidebar_Bot);
         Handle.Tree.Widgets (Nav_About).Text_Content := To_Unbounded_String ("About");
         Handle.Tree.Widgets (Nav_About).Layout.Direction := Dir_Row;

         Nav_Knowledge := Add_Widget (Handle.Tree, Widget_Button, "nav-knowledge", "nav-item", Sidebar_Bot);
         Handle.Tree.Widgets (Nav_Knowledge).Text_Content := To_Unbounded_String ("Knowledge");
         Handle.Tree.Widgets (Nav_Knowledge).Layout.Direction := Dir_Row;

         Nav_Agentic := Add_Widget (Handle.Tree, Widget_Button, "nav-agentic", "nav-item", Sidebar_Bot);
         Handle.Tree.Widgets (Nav_Agentic).Text_Content := To_Unbounded_String ("Agentic");
         Handle.Tree.Widgets (Nav_Agentic).Layout.Direction := Dir_Row;

         --  #main-area (flex: 1, takes remaining space)
         Main_Area_ID := Add_Widget (Handle.Tree, Widget_Panel, "main-area", "main-area", Root_ID);
         Handle.Tree.Widgets (Main_Area_ID).Layout.Direction := Dir_Column;
         Handle.Tree.Widgets (Main_Area_ID).Layout.Flex_Grow := 1.0;

         --  #empty-state (centered greeting, shown when no chat)
         Greeting_ID := Add_Widget (Handle.Tree, Widget_Text, "empty-state", "empty-state", Main_Area_ID);
         Handle.Tree.Widgets (Greeting_ID).Text_Content :=
           To_Unbounded_String ("Heya! I'm Adelaide Zephyrine Charlotte");
         Handle.Tree.Widgets (Greeting_ID).Layout.Direction := Dir_Column;
         Handle.Tree.Widgets (Greeting_ID).Style.Font_Size := 32.0;
         Handle.Tree.Widgets (Greeting_ID).Style.Font_Weight := 500;
      end;

      --  Apply CSS styles to the widget tree
      Apply_CSS_Stylesheet (Handle.Tree, Handle.Stylesheet);

      --  Compute initial layout
      Compute_Layout (Handle.Tree,
                      Float (Config.Width),
                      Float (Config.Height));

      Handle.Is_Initialized := True;
      Handle.Is_Visible := True;
      Handle.Last_Frame_Time := Ada.Real_Time.Clock;

      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:init",
        Message => "Renderer initialized: " &
          Natural'Image (Handle.Tree.Widget_Count) & " widgets, " &
          Natural'Image (Handle.Stylesheet.Rule_Count) & " CSS rules");

      return Handle;
   end Init;

   -- =========================================================================
   -- WINDOW CONTROL
   -- =========================================================================

   procedure Show (Handle : Renderer_Handle) is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;
      Handle.Is_Visible := True;
      Main_Window.Show;
      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:show",
        Message => "Window shown");
   end Show;

   procedure Hide (Handle : Renderer_Handle) is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;
      Handle.Is_Visible := False;
      Main_Window.Hide;
      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:hide",
        Message => "Window hidden");
   end Hide;

   procedure Resize (Handle : Renderer_Handle;
                     Width  : Positive;
                     Height : Positive) is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;
      Handle.Config.Width := Width;
      Handle.Config.Height := Height;

      -- Resize GLFW window
      Main_Window.Set_Size (Glfw.Size (Width), Glfw.Size (Height));

      -- Re-run layout with new dimensions
      Compute_Layout (Handle.Tree, Float (Width), Float (Height));

      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:resize",
        Message => "Resized to " & Natural'Image (Width) & "x" &
          Natural'Image (Height));
   end Resize;

   procedure Close (Handle : in out Renderer_Handle) is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;

      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:close",
        Message => "Shutting down renderer");

      --  Destroy the GLFW window
      --  Citation: GLFW 3.x "glfwDestroyWindow destroys the specified window
      --  and its associated context."
      Main_Window.Destroy;

      --  Terminate GLFW
      --  Citation: GLFW 3.x "glfwTerminate destroys all remaining windows,
      --  frees callbacks, and restores any modified system settings."
      Glfw.Shutdown;

      Handle.Is_Initialized := False;
      Handle.Is_Visible := False;
      Free (Handle);
   end Close;

   -- =========================================================================
   -- CONTENT
   -- =========================================================================

   function Load_CSS (Handle    : Renderer_Handle;
                      File_Path : String)
      return Boolean
   is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return False;
      end if;

      declare
         Parse_Ok : Boolean;
      begin
         Parse_Ok := Parse_CSS_File (File_Path, Handle.Stylesheet);
         if Parse_Ok then
            -- Re-apply CSS to widget tree
            Apply_CSS_Stylesheet (Handle.Tree, Handle.Stylesheet);
            -- Re-compute layout
            Compute_Layout (Handle.Tree,
                            Float (Handle.Config.Width),
                            Float (Handle.Config.Height));
            Adelaide_Trace.Trace_Print (
              Toolcall => "framedisplay:load_css",
              Message => "CSS reloaded: " &
                Natural'Image (Handle.Stylesheet.Rule_Count) & " rules");
         end if;
         return Parse_Ok;
      end;
   end Load_CSS;

   function Get_CSS_Stylesheet (Handle : Renderer_Handle)
      return CSS_Stylesheet
   is
   begin
      if Handle /= null and then Handle.Is_Initialized then
         return Handle.Stylesheet;
      end if;
      return (others => <>);
   end Get_CSS_Stylesheet;

   function Get_Widget_Tree (Handle : Renderer_Handle)
      return Widget_Tree
   is
   begin
      if Handle /= null and then Handle.Is_Initialized then
         return Handle.Tree;
      end if;
      return (others => <>);
   end Get_Widget_Tree;

   procedure Update_Widget_Text (Handle    : Renderer_Handle;
                                 Widget_ID : String;
                                 Text      : String)
   is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;

      -- Find widget by ID and update its text
      declare
         W_Id : constant Widget_ID :=
           Find_Widget_By_ID (Handle.Tree, Widget_ID);
      begin
         if W_Id > 0 then
            Handle.Tree.Widgets (W_Id).Text_Content :=
              To_Unbounded_String (Text);
            Adelaide_Trace.Trace_Print (
              Toolcall => "framedisplay:update_text",
              Message => "Updated widget " & Widget_ID);
         end if;
      end;
   end Update_Widget_Text;

   procedure Execute_Command (Handle  : Renderer_Handle;
                              Command : String)
   is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;

      -- Simple command parser: "action:param1:param2:..."
      -- Split by ':' and dispatch based on the action
      declare
         Cmd : constant String := Command;
         Colon_Pos : Natural := 0;
         Action    : Unbounded_String := Null_Unbounded_String;
         Params    : array (1 .. 4) of Unbounded_String;
         Param_Cnt : Natural := 0;
         Start     : Natural := Cmd'First;
      begin
         -- Parse command and parameters
         for I in Cmd'Range loop
            if Cmd (I) = ':' then
               Param_Cnt := Param_Cnt + 1;
               if Param_Cnt = 1 then
                  Action := To_Unbounded_String (Cmd (Cmd'First .. I - 1));
               elsif Param_Cnt <= 4 then
                  Params (Param_Cnt) := To_Unbounded_String (Cmd (Start .. I - 1));
               end if;
               Start := I + 1;
            end if;
         end loop;
         -- Last parameter
         if Start <= Cmd'Last then
            Param_Cnt := Param_Cnt + 1;
            if Param_Cnt = 1 then
               Action := To_Unbounded_String (Cmd (Start .. Cmd'Last));
            elsif Param_Cnt <= 4 then
               Params (Param_Cnt) := To_Unbounded_String (Cmd (Start .. Cmd'Last));
            end if;
         end if;

         -- Dispatch command
         declare
            Act : constant String := To_String (Action);
         begin
            if Act = "set_visible" and Param_Cnt >= 2 then
               declare
                  W_Id : constant Widget_ID :=
                    Find_Widget_By_ID (Handle.Tree, To_String (Params (1)));
                  Visible : constant Boolean :=
                    To_String (Params (2)) = "true";
               begin
                  if W_Id > 0 then
                     Handle.Tree.Widgets (W_Id).Visible := Visible;
                  end if;
               end;

            elsif Act = "update_text" and Param_Cnt >= 2 then
               Update_Widget_Text (Handle, To_String (Params (1)),
                                   To_String (Params (2)));

            elsif Act = "set_active_tab" and Param_Cnt >= 2 then
               declare
                  W_Id : constant Widget_ID :=
                    Find_Widget_By_ID (Handle.Tree, To_String (Params (1)));
                  Tab_Idx : Natural := 0;
                  Tab_Str : constant String := To_String (Params (2));
               begin
                  for T in Tab_Str'Range loop
                     if Tab_Str (T) >= '0' and then Tab_Str (T) <= '9' then
                        Tab_Idx := Tab_Idx * 10 +
                          (Character'Pos (Tab_Str (T)) - Character'Pos ('0'));
                     end if;
                  end loop;
                  if W_Id > 0 then
                     Handle.Tree.Widgets (W_Id).Active_Tab := Tab_Idx;
                  end if;
               end;

            else
               Adelaide_Trace.Trace_Print (
                 Toolcall => "framedisplay:command",
                 Message => "Unknown command: " & Act);
            end if;
         end;
      end;
   end Execute_Command;

   -- =========================================================================
   -- EVENT PROCESSING
   -- =========================================================================

   function Process_Events (Handle : Renderer_Handle) return Boolean
   is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return False;
      end if;

      --  Calculate delta time for animations
      declare
         Now : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
         Elapsed : constant Ada.Real_Time.Duration :=
           Ada.Real_Time.To_Duration (Now - Handle.Last_Frame_Time);
      begin
         Handle.Delta_Time := Float (Elapsed);
         Handle.Last_Frame_Time := Now;
      end;

      --  Check for window size changes (GLFW polling)
      --  If the user resizes the window, update layout
      declare
         Current_Width  : Glfw.Size;
         Current_Height : Glfw.Size;
      begin
         Main_Window.Get_Size (Current_Width, Current_Height);
         if Natural (Current_Width) /= Handle.Config.Width or
            Natural (Current_Height) /= Handle.Config.Height
         then
            Handle.Config.Width := Natural (Current_Width);
            Handle.Config.Height := Natural (Current_Height);
            Compute_Layout (Handle.Tree,
                            Float (Handle.Config.Width),
                            Float (Handle.Config.Height));
         end if;
      end;

      --  Update animations
      Update_Animations (Handle.Tree, Handle.Delta_Time);

      --  Render frame
      if Handle.Is_Visible then
         Render_Frame (Handle);
      end if;

      --  Poll GLFW events (keyboard, mouse, window close, etc.)
      --  Citation: GLFW 3.x "glfwPollEvents processes all pending events.
      --  This function must be called regularly to process events."
      Glfw.Input.Poll_Events;

      --  Return False if window should close
      --  Citation: GLFW 3.x "glfwWindowShouldClose returns true if the
      --  user has attempted to close the window."
      return not Main_Window.Should_Close;
   end Process_Events;

   procedure Run_Event_Loop (Handle : Renderer_Handle) is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;

      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:event_loop",
        Message => "Entering blocking event loop");

      while Process_Events (Handle) loop
         -- Sleep 10ms between frames to avoid busy-waiting
         -- 16.67ms = 60fps, 10ms ~ 100fps (sufficient for UI)
         delay 0.01;
      end loop;

      Adelaide_Trace.Trace_Print (
        Toolcall => "framedisplay:event_loop",
        Message => "Event loop exited");
   end Run_Event_Loop;

   -- =========================================================================
   -- RENDERING
   -- =========================================================================

   procedure Render_Frame (Handle : Renderer_Handle) is
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;

      --  Make the GLFW OpenGL context current
      --  Citation: GLFW 3.x "glfwMakeContextCurrent makes the OpenGL context
      --  of the specified window current on the calling thread."
      Glfw.Windows.Context.Make_Current (Main_Window'Access);

      --  Set viewport
      --  Citation: OpenGL ES 2.0 §2.12.1 "glViewport sets the viewport
      --  rectangle, which defines the transformation from normalized device
      --  coordinates to window coordinates."
      GL.Window.Set_Viewport (0, 0,
                              GL.Types.Int (Handle.Config.Width),
                              GL.Types.Size (Handle.Config.Height));

      --  Render the widget tree (this issues all OpenGL draw calls)
      Render_Tree (Handle.Tree);

      --  Swap front and back buffers
      --  Citation: GLFW 3.x "glfwSwapBuffers swaps the front and back
      --  buffers of the specified window."
      Glfw.Windows.Context.Swap_Buffers (Main_Window'Access);
   end Render_Frame;

   -- =========================================================================
   -- SPLASH SCREEN
   -- =========================================================================

   procedure Set_Opacity (Handle  : Renderer_Handle;
                          Opacity : Float)
   is
      Clamped : constant Float := Float'Max (0.0, Float'Min (1.0, Opacity));
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;
      Handle.Current_Opacity := Clamped;
   end Set_Opacity;

   procedure Fade_In (Handle   : Renderer_Handle;
                      Duration : Float := 1.0)
   is
      Start_Time : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
      Elapsed    : Float := 0.0;
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;

      Handle.Is_Visible := True;

      while Elapsed < Duration loop
         -- Linear fade from 0.0 to 1.0
         Handle.Current_Opacity := Elapsed / Duration;
         Render_Frame (Handle);
         delay 0.016;  -- ~60fps
         declare
            Now : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
         begin
            Elapsed := Float (Ada.Real_Time.To_Duration (Now - Start_Time));
         end;
      end loop;

      Handle.Current_Opacity := 1.0;
   end Fade_In;

   procedure Fade_Out (Handle   : Renderer_Handle;
                       Duration : Float := 0.5)
   is
      Start_Time : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
      Elapsed    : Float := 0.0;
   begin
      if Handle = null or else not Handle.Is_Initialized then
         return;
      end if;

      while Elapsed < Duration loop
         -- Linear fade from 1.0 to 0.0
         Handle.Current_Opacity := 1.0 - (Elapsed / Duration);
         Render_Frame (Handle);
         delay 0.016;  -- ~60fps
         declare
            Now : constant Ada.Real_Time.Time := Ada.Real_Time.Clock;
         begin
            Elapsed := Float (Ada.Real_Time.To_Duration (Now - Start_Time));
         end;
      end loop;

      Handle.Current_Opacity := 0.0;
      Handle.Is_Visible := False;
   end Fade_Out;

end Zephyrine_Main_Framedisplay;
