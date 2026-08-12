pragma SPARK_Mode (Off);
-- ============================================================================
-- ZEPHYRINE_WIDGET_TREE — Widget tree implementation
-- ============================================================================
--
-- AXIOMS AND CITATIONS:
--   - Widget tree construction follows the Composite pattern (GoF, 1994)
--     Section 4.3: Composite — "Compose objects into tree structures to
--     represent part-whole hierarchies"
--   - Layout algorithm: Simplified CSS Box Model (W3C CSS Box Model Level 3)
--     Content area = allocated space - padding - border
--     Border area = content + padding
--     Margin area = border + margin
--   - Flexbox layout: Simplified implementation of CSS Flexbox Level 1
--     https://www.w3.org/TR/css-flexbox-1/
--     Main axis = direction (column or row), Cross axis = perpendicular
--   - Rendering: Depth-first traversal, back-to-front for correct overlap
--
-- ============================================================================

with Ada.Text_IO;
with Adelaide_Trace;

package body Zephyrine_Widget_Tree is

   -- =========================================================================
   -- TREE CONSTRUCTION
   -- =========================================================================

   procedure Init_Tree (Tree : in out Widget_Tree) is
   begin
      Tree.Widget_Count := 1;
      Tree.Root_ID := 1;
      Tree.Widgets (1).ID := 1;
      Tree.Widgets (1).Kind := Widget_Container;
      Tree.Widgets (1).Tag := (Raw => To_Unbounded_String ("root"));
      Tree.Widgets (1).Class := (Raw => Null_Unbounded_String);
      Tree.Widgets (1).Parent := 0;
      Tree.Widgets (1).Visible := True;
      Tree.Widgets (1).Style.Background_Color := (0.043, 0.047, 0.055, 1.0); -- #0b0c0e
   end Init_Tree;

   function Add_Widget
     (Tree      : in out Widget_Tree;
      Kind      : Widget_Kind;
      Tag       : String;
      Class     : String;
      Parent_ID : Widget_ID := 0)
      return Widget_ID
   is
      New_ID : Widget_ID;
   begin
      if Tree.Widget_Count >= Max_Widgets then
         Adelaide_Trace.Trace_Print (
           Toolcall => "widget_tree:add",
           Message => "ERROR: Widget limit reached (" &
             Natural'Image (Max_Widgets) & ")");
         return 0;
      end if;

      Tree.Widget_Count := Tree.Widget_Count + 1;
      New_ID := Tree.Widget_Count;

      -- Initialize the new widget
      Tree.Widgets (New_ID).ID := New_ID;
      Tree.Widgets (New_ID).Kind := Kind;
      Tree.Widgets (New_ID).Tag := (Raw => To_Unbounded_String (Tag));
      Tree.Widgets (New_ID).Class := (Raw => To_Unbounded_String (Class));
      Tree.Widgets (New_ID).Parent := Parent_ID;
      Tree.Widgets (New_ID).Visible := True;
      Tree.Widgets (New_ID).Child_Count := 0;

      -- Set sensible defaults based on widget kind
      case Kind is
         when Widget_Panel =>
            Tree.Widgets (New_ID).Layout.Direction := Dir_Column;
            Tree.Widgets (New_ID).Layout.Overflow := Overflow_Hidden;

         when Widget_Button =>
            Tree.Widgets (New_ID).Layout.Direction := Dir_Row;
            Tree.Widgets (New_ID).Style.Background_Color :=
              (1.0, 1.0, 1.0, 0.06);  -- rgba(255,255,255,0.06)

         when Widget_Text =>
            Tree.Widgets (New_ID).Style.Text_Color :=
              (0.89, 0.89, 0.89, 1.0);  -- #e3e3e3
            Tree.Widgets (New_ID).Style.Font_Size := 16.0;

         when Widget_Input =>
            Tree.Widgets (New_ID).Layout.Direction := Dir_Row;

         when Widget_Scrollable =>
            Tree.Widgets (New_ID).Layout.Overflow := Overflow_Scroll;

         when Widget_Tab =>
            Tree.Widgets (New_ID).Layout.Direction := Dir_Column;

         when Widget_Image =>
            null;

         when Widget_Background =>
            Tree.Widgets (New_ID).Position := Pos_Absolute;
            Tree.Widgets (New_ID).Layout.Position := Pos_Absolute;

         when Widget_Container =>
            null;
      end case;

      -- Add to parent's children list
      if Parent_ID > 0 and then Parent_ID <= Tree.Widget_Count then
         if Tree.Widgets (Parent_ID).Child_Count < Max_Children then
            Tree.Widgets (Parent_ID).Child_Count :=
              Tree.Widgets (Parent_ID).Child_Count + 1;
            Tree.Widgets (Parent_ID).Children (
              Tree.Widgets (Parent_ID).Child_Count) := New_ID;
         end if;
      end if;

      return New_ID;
   end Add_Widget;

   procedure Remove_Widget
     (Tree : in out Widget_Tree;
      ID   : Widget_ID)
   is
   begin
      if ID = 0 or else ID > Tree.Widget_Count then
         return;
      end if;

      -- Remove from parent's children list
      declare
         Parent_ID : constant Widget_ID := Tree.Widgets (ID).Parent;
      begin
         if Parent_ID > 0 and then Parent_ID <= Tree.Widget_Count then
            for C in 1 .. Tree.Widgets (Parent_ID).Child_Count loop
               if Tree.Widgets (Parent_ID).Children (C) = ID then
                  -- Shift remaining children left
                  for J in C .. Tree.Widgets (Parent_ID).Child_Count - 1 loop
                     Tree.Widgets (Parent_ID).Children (J) :=
                       Tree.Widgets (Parent_ID).Children (J + 1);
                  end loop;
                  Tree.Widgets (Parent_ID).Child_Count :=
                    Tree.Widgets (Parent_ID).Child_Count - 1;
                  exit;
               end if;
            end loop;
         end;
      end;

      -- Mark as removed (set Child_Count to 0, Kind to a sentinel)
      Tree.Widgets (ID).Child_Count := 0;
      Tree.Widgets (ID).Visible := False;
   end Remove_Widget;

   function Find_Widget_By_ID
     (Tree       : Widget_Tree;
      Search_ID  : String)
      return Widget_ID
   is
   begin
      for I in 1 .. Tree.Widget_Count loop
         if To_String (Tree.Widgets (I).Tag.Raw) = Search_ID then
            return Tree.Widgets (I).ID;
         end if;
      end loop;
      return 0;
   end Find_Widget_By_ID;

   function Find_Widget_By_Class
     (Tree         : Widget_Tree;
      Search_Class : String)
      return Widget_ID
   is
      Class_Str : constant String := Search_Class;
   begin
      for I in 1 .. Tree.Widget_Count loop
         declare
            Wc : constant String := To_String (Tree.Widgets (I).Class.Raw);
         begin
            -- Simple substring check (for "nav-item" matching "nav-item active")
            if Wc'Length >= Class_Str'Length then
               for J in Wc'First .. Wc'Last - Class_Str'Length + 1 loop
                  if Wc (J .. J + Class_Str'Length - 1) = Class_Str then
                     return Tree.Widgets (I).ID;
                  end if;
               end loop;
            end if;
         end;
      end loop;
      return 0;
   end Find_Widget_By_Class;

   -- =========================================================================
   -- CSS STYLE APPLICATION
   -- =========================================================================

   procedure Apply_CSS_Stylesheet
     (Tree       : in out Widget_Tree;
      Stylesheet : CSS_Stylesheet)
   is
      --  Map widget tag/class to CSS selector for lookup.
      --  The selector is formed as "#tag" for ID-based or ".class" for class-based.
      function Widget_To_Selector (W : Widget) return String is
         Tag_Str  : constant String := To_String (W.Tag.Raw);
         Class_Str : constant String := To_String (W.Class.Raw);
      begin
         -- Try class-based selector first (e.g. ".nav-item")
         if Class_Str'Length > 0 then
            -- Take the first class name (before space)
            for I in Class_Str'Range loop
               if Class_Str (I) = ' ' then
                  return "." & Class_Str (Class_Str'First .. I - 1);
               end if;
            end loop;
            return "." & Class_Str;
         end if;

         -- Fall back to tag-based selector (e.g. "#sidebar")
         return "#" & Tag_Str;
      end Widget_To_Selector;

      procedure Apply_To_Widget (W_Id : Widget_ID) is
         W     : Widget renames Tree.Widgets (W_Id);
         Sel   : constant String := Widget_To_Selector (W);
         Result : CSS_Lookup_Result;
      begin
         -- Apply background-color
         Result := Lookup_Property (Stylesheet, Sel, Prop_Background_Color);
         if Result.Found and then Result.Value.Tag = Tag_Color then
            W.Style.Background_Color := Result.Value.Color;
         end if;

         -- Apply color (text color)
         Result := Lookup_Property (Stylesheet, Sel, Prop_Color);
         if Result.Found and then Result.Value.Tag = Tag_Color then
            W.Style.Text_Color := Result.Value.Color;
         end if;

         -- Apply font-size
         Result := Lookup_Property (Stylesheet, Sel, Prop_Font_Size);
         if Result.Found and then Result.Value.Tag = Tag_Length then
            W.Style.Font_Size := Length_To_Pixels (Result.Value.Length);
         end if;

         -- Apply font-weight
         Result := Lookup_Property (Stylesheet, Sel, Prop_Font_Weight);
         if Result.Found and then Result.Value.Tag = Tag_Number then
            W.Style.Font_Weight := Natural (Result.Value.Num);
         end if;

         -- Apply border-radius
         Result := Lookup_Property (Stylesheet, Sel, Prop_Border_Radius);
         if Result.Found and then Result.Value.Tag = Tag_Length then
            W.Style.Border_Radius := Length_To_Pixels (Result.Value.Length);
         end if;

         -- Apply opacity
         Result := Lookup_Property (Stylesheet, Sel, Prop_Opacity);
         if Result.Found and then Result.Value.Tag = Tag_Number then
            W.Style.Opacity := Result.Value.Num;
         end if;

         -- Apply padding (simplified: use as uniform padding)
         Result := Lookup_Property (Stylesheet, Sel, Prop_Padding);
         if Result.Found and then Result.Value.Tag = Tag_Length then
            declare
               Pad : constant Float := Length_To_Pixels (Result.Value.Length);
            begin
               W.Layout.Padding := (Pad, Pad, Pad, Pad);
            end;
         end if;

         -- Apply margin
         Result := Lookup_Property (Stylesheet, Sel, Prop_Margin);
         if Result.Found and then Result.Value.Tag = Tag_Length then
            declare
               Mar : constant Float := Length_To_Pixels (Result.Value.Length);
            begin
               W.Layout.Margin := (Mar, Mar, Mar, Mar);
            end;
         end if;

         -- Apply width/height
         Result := Lookup_Property (Stylesheet, Sel, Prop_Width);
         if Result.Found and then Result.Value.Tag = Tag_Length then
            W.Geometry.Content_Box.Width := Length_To_Pixels (Result.Value.Length);
         end if;

         Result := Lookup_Property (Stylesheet, Sel, Prop_Height);
         if Result.Found and then Result.Value.Tag = Tag_Length then
            W.Geometry.Content_Box.Height := Length_To_Pixels (Result.Value.Length);
         end if;

         -- Apply display (for visibility)
         Result := Lookup_Property (Stylesheet, Sel, Prop_Display);
         if Result.Found and then Result.Value.Tag = Tag_Keyword then
            declare
               Disp : constant String := To_String (Result.Value.Keyword);
            begin
               if Disp = "none" then
                  W.Visible := False;
               elsif Disp = "flex" then
                  W.Layout.Direction := Dir_Column;  -- Default flex direction
               end if;
            end;
         end if;

         -- Apply flex-direction
         Result := Lookup_Property (Stylesheet, Sel, Prop_Flex_Direction);
         if Result.Found and then Result.Value.Tag = Tag_Keyword then
            declare
               Flex_D : constant String := To_String (Result.Value.Keyword);
            begin
               if Flex_D = "row" then
                  W.Layout.Direction := Dir_Row;
               elsif Flex_D = "column" then
                  W.Layout.Direction := Dir_Column;
               end if;
            end;
         end if;

         -- Apply overflow
         Result := Lookup_Property (Stylesheet, Sel, Prop_Overflow);
         if Result.Found and then Result.Value.Tag = Tag_Keyword then
            declare
               Ovf : constant String := To_String (Result.Value.Keyword);
            begin
               if Ovf = "hidden" then
                  W.Layout.Overflow := Overflow_Hidden;
               elsif Ovf = "scroll" or else Ovf = "auto" then
                  W.Layout.Overflow := Overflow_Scroll;
               elsif Ovf = "visible" then
                  W.Layout.Overflow := Overflow_Visible;
               end if;
            end;
         end if;

         -- Apply gap
         Result := Lookup_Property (Stylesheet, Sel, Prop_Gap);
         if Result.Found and then Result.Value.Tag = Tag_Length then
            W.Layout.Gap := Length_To_Pixels (Result.Value.Length);
         end if;

         -- Apply position
         Result := Lookup_Property (Stylesheet, Sel, Prop_Position);
         if Result.Found and then Result.Value.Tag = Tag_Keyword then
            declare
               Pos_Str : constant String := To_String (Result.Value.Keyword);
            begin
               if Pos_Str = "absolute" then
                  W.Layout.Position := Pos_Absolute;
               elsif Pos_Str = "fixed" then
                  W.Layout.Position := Pos_Fixed;
               elsif Pos_Str = "relative" then
                  W.Layout.Position := Pos_Relative;
               end if;
            end;
         end if;

         -- Apply z-index
         Result := Lookup_Property (Stylesheet, Sel, Prop_Z_Index);
         if Result.Found and then Result.Value.Tag = Tag_Number then
            -- Store as a simple integer for now (no z-index in GL)
            null;
         end if;

         -- Apply box-shadow (simplified: just the color and blur)
         Result := Lookup_Property (Stylesheet, Sel, Prop_Box_Shadow);
         if Result.Found then
            W.Style.Box_Shadow := (0.0, 0.0, 0.0, 0.4);
            W.Style.Box_Shadow_Blur := 20.0;
         end if;

         -- Apply animation
         Result := Lookup_Property (Stylesheet, Sel, Prop_Animations);
         if Result.Found and then Result.Value.Tag = Tag_Keyword then
            declare
               Anim_Name : constant String := To_String (Result.Value.Keyword);
            begin
               -- Map animation name to Animation_Kind
               if Anim_Name = "pulseOrb" then
                  W.Animation.Kind := Anim_Pulse;
                  W.Animation.Duration := 12.0;
                  W.Animation.Is_Active := True;
               elsif Anim_Name = "twinkle" then
                  W.Animation.Kind := Anim_Twinkle;
                  W.Animation.Duration := 3.0;
                  W.Animation.Is_Active := True;
               elsif Anim_Name = "shootingStar" then
                  W.Animation.Kind := Anim_Shooting_Star;
                  W.Animation.Duration := 12.0;
                  W.Animation.Is_Active := True;
               elsif Anim_Name = "fadeIn" then
                  W.Animation.Kind := Anim_Fade_In;
                  W.Animation.Duration := 1.0;
                  W.Animation.Is_Active := True;
               elsif Anim_Name = "spinGradient" then
                  W.Animation.Kind := Anim_Spin_Gradient;
                  W.Animation.Duration := 4.0;
                  W.Animation.Is_Active := True;
               elsif Anim_Name = "floatLogo" then
                  W.Animation.Kind := Anim_Float_Logo;
                  W.Animation.Duration := 6.0;
                  W.Animation.Is_Active := True;
               end if;
            end;
         end if;

         -- Apply animation shorthand: "animation: name duration timing delay count"
         -- (Already handled above via Prop_Animations)

         -- Recursively apply to children
         for C in 1 .. W.Child_Count loop
            Apply_To_Widget (W.Children (C));
         end loop;
      end Apply_To_Widget;

   begin
      -- Apply styles starting from root
      Apply_To_Widget (Tree.Root_ID);

      Adelaide_Trace.Trace_Print (
        Toolcall => "widget_tree:apply_css",
        Message => "Applied CSS to " & Natural'Image (Tree.Widget_Count) &
          " widgets");
   end Apply_CSS_Stylesheet;

   -- =========================================================================
   -- LAYOUT ENGINE
   -- =========================================================================

   procedure Compute_Layout
     (Tree       : in out Widget_Tree;
      Root_Width : Float;
      Root_Height: Float)
   is
      procedure Layout_Widget (W_Id : Widget_ID; Box : Rect) is
         W : Widget renames Tree.Widgets (W_Id);
         Content_X : Float := Box.X + W.Layout.Margin.Left + W.Layout.Border_Width.Left + W.Layout.Padding.Left;
         Content_Y : Float := Box.Y + W.Layout.Margin.Top + W.Layout.Border_Width.Top + W.Layout.Padding.Top;
         Content_W : Float := Box.Width - W.Layout.Margin.Left - W.Layout.Margin.Right
           - W.Layout.Border_Width.Left - W.Layout.Border_Width.Right
           - W.Layout.Padding.Left - W.Layout.Padding.Right;
         Content_H : Float := Box.Height - W.Layout.Margin.Top - W.Layout.Margin.Bottom
           - W.Layout.Border_Width.Top - W.Layout.Border_Width.Bottom
           - W.Layout.Padding.Top - W.Layout.Padding.Bottom;
         Child_Y   : Float := Content_Y;
         Child_X   : Float := Content_X;
         Available_W : Float := Content_W;
         Available_H : Float := Content_H;

         -- Count children and total flex-grow
         Total_Flex : Float := 0.0;
         Non_Flex_H : Float := 0.0;
      begin
         -- Set geometry
         W.Geometry.Margin_Box := Box;
         W.Geometry.Border_Box := (
           X      => Box.X + W.Layout.Margin.Left,
           Y      => Box.Y + W.Layout.Margin.Top,
           Width  => Box.Width - W.Layout.Margin.Left - W.Layout.Margin.Right,
           Height => Box.Height - W.Layout.Margin.Top - W.Layout.Margin.Bottom
         );
         W.Geometry.Content_Box := (
           X      => Content_X,
           Y      => Content_Y,
           Width  => Content_W,
           Height => Content_H
         );

         -- Count flex items
         for C in 1 .. W.Child_Count loop
            declare
               Child : Widget renames Tree.Widgets (W.Children (C));
            begin
               if Child.Visible then
                  Total_Flex := Total_Flex + Child.Layout.Flex_Grow;
               end if;
            end;
         end loop;

         -- Layout children
         if W.Layout.Direction = Dir_Column then
            -- Vertical layout: children stacked top to bottom
            for C in 1 .. W.Child_Count loop
               declare
                  Child : Widget renames Tree.Widgets (W.Children (C));
                  Child_Box : Rect;
               begin
                  if Child.Visible then
                     Child_Box := (
                        X      => Child_X,
                        Y      => Child_Y + Child.Layout.Margin.Top,
                        Width  => Available_W - Child.Layout.Margin.Left - Child.Layout.Margin.Right,
                        Height => (if Child.Layout.Flex_Grow > 0.0
                                   then 0.0  -- Will be distributed below
                                   else Content_H - Child.Layout.Margin.Top - Child.Layout.Margin.Bottom)
                     );

                     if Child.Layout.Flex_Grow > 0.0 and then Total_Flex > 0.0 then
                        Child_Box.Height := Content_H * (Child.Layout.Flex_Grow / Total_Flex);
                     end if;

                     Layout_Widget (Child.ID, Child_Box);
                     Child_Y := Child_Y + Child.Box.Height + Child.Layout.Margin.Top + Child.Layout.Margin.Bottom + W.Layout.Gap;
                  end if;
               end;
            end loop;

         elsif W.Layout.Direction = Dir_Row then
            -- Horizontal layout: children side by side
            for C in 1 .. W.Child_Count loop
               declare
                  Child : Widget renames Tree.Widgets (W.Children (C));
                  Child_Box : Rect;
               begin
                  if Child.Visible then
                     Child_Box := (
                        X      => Child_X + Child.Layout.Margin.Left,
                        Y      => Child_Y,
                        Width  => (if Child.Layout.Flex_Grow > 0.0
                                   then 0.0
                                   else Content_W - Child.Layout.Margin.Left - Child.Layout.Margin.Right),
                        Height => Available_H - Child.Layout.Margin.Top - Child.Layout.Margin.Bottom
                     );

                     if Child.Layout.Flex_Grow > 0.0 and then Total_Flex > 0.0 then
                        Child_Box.Width := Content_W * (Child.Layout.Flex_Grow / Total_Flex);
                     end if;

                     Layout_Widget (Child.ID, Child_Box);
                     Child_X := Child_X + Child_Box.Width + Child.Layout.Margin.Left + Child.Layout.Margin.Right + W.Layout.Gap;
                  end if;
               end;
            end loop;
         end if;
      end Layout_Widget;

   begin
      -- Start layout from root with full window dimensions
      declare
         Root_Box : constant Rect := (
            X      => 0.0,
            Y      => 0.0,
            Width  => Root_Width,
            Height => Root_Height
         );
      begin
         Layout_Widget (Tree.Root_ID, Root_Box);
      end;

      Adelaide_Trace.Trace_Print (
        Toolcall => "widget_tree:layout",
        Message => "Layout computed: " & Natural'Image (Tree.Widget_Count) &
          " widgets, " & Float'Image (Root_Width) & "x" &
          Float'Image (Root_Height));
   end Compute_Layout;

   -- =========================================================================
   -- RENDERING — OpenGL ES 2.0 draw calls via OpenGLAda
   -- =========================================================================
   --
   -- AXIOMS AND CITATIONS:
   --   - OpenGLAda (flyx/OpenGLAda) v0.9.0: Thick Ada binding for OpenGL
   --     Source: https://github.com/flyx/OpenGLAda
   --     Provides type-safe GL objects (Shader, Program, Buffer) with
   --     reference counting and automatic resource management.
   --   - OpenGL ES 2.0 §3.5-3.6: Shader compilation and program linking
   --   - OpenGL ES 2.0 §3.8: Drawing primitives (glDrawArrays)
   --   - OpenGL ES Shading Language 1.00 §4.1.9: gl_Position output
   --
   -- IMPLEMENTATION NOTES:
   --   - Uses OpenGLAda's typed GL objects instead of raw GLuint handles
   --   - GL.Objects.Shaders.Shader for vertex/fragment shaders
   --   - GL.Objects.Programs.Program for linked shader programs
   --   - GL.Objects.Buffers.Buffer for vertex buffer objects (VBOs)
   --   - GLSL ES 1.00 shaders for ES 2.0 compatibility
   --   - Orthographic projection maps pixel coords to clip space
   --

   with GL;
   with GL.Types;
   with GL.Types.Singles;
   with GL.Objects;
   with GL.Objects.Shaders;
   with GL.Objects.Programs;
   with GL.Objects.Buffers;
   with GL.Attributes;
   with GL.Uniforms;
   with GL.Toggles;
   with GL.Blending;
   with GL.Buffers;
   with GL.Drawing;

   use GL;
   use GL.Types;
   use GL.Types.Singles;
   use GL.Objects;
   use GL.Objects.Shaders;
   use GL.Objects.Programs;
   use GL.Objects.Buffers;
   use GL.Attributes;
   use GL.Uniforms;
   use GL.Toggles;
   use GL.Blending;

   --  Internal state for the renderer (shader programs, VBOs, etc.)
   --  Initialized once at startup, reused for all draw calls.
   --  Uses OpenGLAda's reference-counted GL objects for automatic cleanup.
   type Render_State is record
      Initialized      : Boolean := False;
      Rect_Program     : GL.Objects.Programs.Program;  -- Shader program for solid-color quads
      Text_Program     : GL.Objects.Programs.Program;  -- Shader program for text (future)
      Rect_VBO         : GL.Objects.Buffers.Buffer;    -- Vertex buffer for unit quad
      Proj_Matrix      : GL.Types.Singles.Matrix4 := (others => (others => 0.0));
   end record;

   --  Global render state (initialized on first render)
   G_Render_State : Render_State;

   --  Unit quad vertices: position (x,y) + color (r,g,b,a)
   --  This is a 1x1 quad that gets scaled via uniform matrix.
   --  6 vertices × 4 floats = 24 floats total.
   Unit_Quad_Data : constant GL.Types.Singles.Vector24 :=
     (0.0, 0.0,  1.0, 1.0, 1.0, 1.0,  -- bottom-left
      1.0, 0.0,  1.0, 1.0, 1.0, 1.0,  -- bottom-right
      0.0, 1.0,  1.0, 1.0, 1.0, 1.0,  -- top-left
      1.0, 0.0,  1.0, 1.0, 1.0, 1.0,  -- bottom-right
      1.0, 1.0,  1.0, 1.0, 1.0, 1.0,  -- top-right
      0.0, 1.0,  1.0, 1.0, 1.0, 1.0); -- top-left

   --  GLSL ES 1.00 vertex shader for solid-color quads.
   --  Citation: OpenGL ES Shading Language 1.00 §4.1.9 (gl_Position)
   Quad_Vertex_Shader : constant String :=
     "#version 100" & ASCII.LF &
     "precision mediump float;" & ASCII.LF &
     "attribute vec2 a_Position;" & ASCII.LF &
     "attribute vec4 a_Color;" & ASCII.LF &
     "uniform mat4 u_Projection;" & ASCII.LF &
     "uniform mat4 u_Model;" & ASCII.LF &
     "varying vec4 v_Color;" & ASCII.LF &
     "void main() {" & ASCII.LF &
     "    gl_Position = u_Projection * u_Model * vec4(a_Position, 0.0, 1.0);" & ASCII.LF &
     "    v_Color = a_Color;" & ASCII.LF &
     "}" & ASCII.LF;

   --  GLSL ES 1.00 fragment shader for solid-color quads.
   Quad_Fragment_Shader : constant String :=
     "#version 100" & ASCII.LF &
     "precision mediump float;" & ASCII.LF &
     "varying vec4 v_Color;" & ASCII.LF &
     "void main() {" & ASCII.LF &
     "    gl_FragColor = v_Color;" & ASCII.LF &
     "}" & ASCII.LF;

   --  Helper: Compile a shader and check for errors.
   --  Uses OpenGLAda's Shader type with Initialize_Id + Set_Source + Compile.
   --  Citation: OpenGLAda API — GL.Objects.Shaders
   function Compile_Shader_Checked (Source       : String;
                                     Shader_Kind  : Shader_Type)
      return GL.Objects.Shaders.Shader
   is
      Shader : GL.Objects.Shaders.Shader (Kind => Shader_Kind);
   begin
      Shader.Initialize_Id;
      Shader.Set_Source (Source);
      Shader.Compile;

      if not Shader.Compile_Status then
         declare
            Log : constant String := Shader.Info_Log;
         begin
            if Log'Length > 0 then
               Adelaide_Trace.Trace_Print (
                 Toolcall => "renderer:compile_shader",
                 Message => "SHADER ERROR: " & Log);
            end if;
         end;
         Shader.Clear;
         return Shader;  -- Caller checks Compile_Status
      end if;

      return Shader;
   end Compile_Shader_Checked;

   --  Helper: Link a shader program.
   --  Uses OpenGLAda's Program type with Initialize_Id + Attach + Link.
   --  Citation: OpenGLAda API — GL.Objects.Programs
   function Link_Program_Checked (Vert_Shader, Frag_Shader : GL.Objects.Shaders.Shader)
      return GL.Objects.Programs.Program
   is
      Prog : GL.Objects.Programs.Program;
   begin
      Prog.Initialize_Id;
      Prog.Attach (Vert_Shader);
      Prog.Attach (Frag_Shader);
      Prog.Link;

      if not Prog.Link_Status then
         declare
            Log : constant String := Prog.Info_Log;
         begin
            if Log'Length > 0 then
               Adelaide_Trace.Trace_Print (
                 Toolcall => "renderer:link_program",
                 Message => "LINK ERROR: " & Log);
            end if;
         end;
         Prog.Clear;
      end if;

      return Prog;
   end Link_Program_Checked;

   --  Initialize the renderer: compile shaders, create VBOs.
   --  Uses OpenGLAda's typed API for all GL operations.
   --  Citation: OpenGLAda API — GL.Objects.Shaders, GL.Objects.Programs,
   --            GL.Objects.Buffers, GL.Toggles, GL.Blending
   procedure Init_Renderer (Width, Height : Float) is
      Vert_Shader : GL.Objects.Shaders.Shader
        (Kind => GL.Objects.Shaders.Vertex_Shader);
      Frag_Shader : GL.Objects.Shaders.Shader
        (Kind => GL.Objects.Shaders.Fragment_Shader);
   begin
      if G_Render_State.Initialized then
         return;
      end if;

      Adelaide_Trace.Trace_Print (
        Toolcall => "renderer:init",
        Message => "Initializing OpenGL ES 2.0 renderer via OpenGLAda");

      -- Enable blending for transparency
      -- Citation: OpenGL ES 2.0 §3.7.7 — glBlendFunc
      GL.Toggles.Enable (GL.Toggles.Blend);
      GL.Blending.Set_Blend_Func (GL.Blending.Src_Alpha, GL.Blending.One_Minus_Src_Alpha);

      -- Compile shaders using OpenGLAda's typed Shader objects
      Vert_Shader := Compile_Shader_Checked (Quad_Vertex_Shader,
        GL.Objects.Shaders.Vertex_Shader);
      Frag_Shader := Compile_Shader_Checked (Quad_Fragment_Shader,
        GL.Objects.Shaders.Fragment_Shader);

      if not Vert_Shader.Compile_Status or else not Frag_Shader.Compile_Status then
         Adelaide_Trace.Trace_Print (
           Toolcall => "renderer:init",
           Message => "ERROR: Shader compilation failed");
         return;
      end if;

      -- Link program using OpenGLAda's typed Program object
      G_Render_State.Rect_Program := Link_Program_Checked (Vert_Shader, Frag_Shader);

      -- Shaders can be cleared after linking (program retains them)
      Vert_Shader.Clear;
      Frag_Shader.Clear;

      if not G_Render_State.Rect_Program.Initialized then
         Adelaide_Trace.Trace_Print (
           Toolcall => "renderer:init",
           Message => "ERROR: Program linking failed");
         return;
      end if;

      -- Create VBO for unit quad using OpenGLAda's Buffer type
      G_Render_State.Rect_VBO.Initialize_Id;
      GL.Objects.Buffers.Bind (GL.Objects.Buffers.Array_Buffer,
                               G_Render_State.Rect_VBO);
      GL.Objects.Buffers.Allocate (GL.Objects.Buffers.Array_Buffer,
                                   Unit_Quad_Data'Size / 8,
                                   GL.Objects.Buffers.Static_Draw);

      -- Set up orthographic projection matrix
      -- Maps [0,Width] x [0,Height] to [-1,1] x [-1,1]
      -- Citation: OpenGL ES 2.0 §4.1.9 — orthographic projection
      G_Render_State.Proj_Matrix := (others => (others => 0.0));
      G_Render_State.Proj_Matrix (1, 1) := Single (2.0 / Width);   -- Scale X
      G_Render_State.Proj_Matrix (2, 2) := Single (-2.0 / Height); -- Scale Y (flip Y)
      G_Render_State.Proj_Matrix (3, 3) := -1.0;                   -- Z depth
      G_Render_State.Proj_Matrix (4, 1) := -1.0;                   -- Translate X to center
      G_Render_State.Proj_Matrix (4, 2) := 1.0;                    -- Translate Y to center
      G_Render_State.Proj_Matrix (4, 4) := 1.0;                    -- W homogeneous

      G_Render_State.Initialized := True;

      Adelaide_Trace.Trace_Print (
        Toolcall => "renderer:init",
        Message => "Renderer initialized successfully via OpenGLAda");
   end Init_Renderer;

   --  Draw a filled rectangle at (X, Y) with given Width, Height, and Color.
   --  Uses OpenGLAda's typed uniform/attribute API.
   --  Citation: OpenGL ES 2.0 §3.5 — glUniform, §3.6 — glVertexAttribPointer
   procedure Draw_Filled_Rect (X, Y, W, H : Float;
                               R, G, B, A : Float)
   is
      Loc_Proj    : GL.Objects.Programs.Uniforms.Uniform;
      Loc_Model   : GL.Objects.Programs.Uniforms.Uniform;
      Loc_Pos     : GL.Objects.Programs.Attributes.Attribute;
      Loc_Color   : GL.Objects.Programs.Attributes.Attribute;
      Model_Matrix : GL.Types.Singles.Matrix4 := (others => (others => 0.0));
   begin
      if not G_Render_State.Rect_Program.Initialized then
         return;
      end if;

      G_Render_State.Rect_Program.Use_Program;

      -- Model matrix: scale and translate the unit quad
      Model_Matrix (1, 1) := Single (W);   -- Scale X
      Model_Matrix (2, 2) := Single (H);   -- Scale Y
      Model_Matrix (3, 3) := 1.0;          -- Scale Z
      Model_Matrix (4, 1) := Single (X);   -- Translate X
      Model_Matrix (4, 2) := Single (Y);   -- Translate Y
      Model_Matrix (4, 4) := 1.0;          -- W

      -- Get uniform and attribute locations
      Loc_Proj  := G_Render_State.Rect_Program.Uniform_Location ("u_Projection");
      Loc_Model := G_Render_State.Rect_Program.Uniform_Location ("u_Model");
      Loc_Pos   := G_Render_State.Rect_Program.Attrib_Location ("a_Position");
      Loc_Color := G_Render_State.Rect_Program.Attrib_Location ("a_Color");

      -- Set uniforms
      GL.Uniforms.Set_Single (Loc_Proj, G_Render_State.Proj_Matrix);
      GL.Uniforms.Set_Single (Loc_Model, Model_Matrix);

      -- Set vertex color via uniform (overrides vertex attribute)
      GL.Uniforms.Set_Single (
        G_Render_State.Rect_Program.Uniform_Location ("u_Color"),
        Single (R), Single (G), Single (B), Single (A));

      -- Bind VBO and set vertex attributes
      GL.Objects.Buffers.Bind (GL.Objects.Buffers.Array_Buffer,
                               G_Render_State.Rect_VBO);
      GL.Attributes.Enable_Vertex_Attrib_Array (Loc_Pos);
      GL.Attributes.Enable_Vertex_Attrib_Array (Loc_Color);
      GL.Attributes.Set_Vertex_Attrib_Pointer (
        Loc_Pos, 2, GL.Types.Single_Type, False,
        6 * 4,  -- stride: 6 floats * 4 bytes
        0);     -- offset 0
      GL.Attributes.Set_Vertex_Attrib_Pointer (
        Loc_Color, 4, GL.Types.Single_Type, False,
        6 * 4,  -- stride: 6 floats * 4 bytes
        8);     -- offset: 2 floats * 4 bytes

      -- Draw the quad (2 triangles, 6 vertices)
      GL.Drawing.Draw_Arrays (GL.Types.Triangles, 0, 6);

      -- Cleanup
      GL.Attributes.Disable_Vertex_Attrib_Array (Loc_Pos);
      GL.Attributes.Disable_Vertex_Attrib_Array (Loc_Color);
      GL.Objects.Programs.Program'(G_Render_State.Rect_Program).Use_Program;
   end Draw_Filled_Rect;

   --  Render a single widget and its children (depth-first traversal).
   --  For each visible widget: draw background, draw border, recurse children.
   procedure Render_Widget (Tree : Widget_Tree; W_Id : Widget_ID) is
      W : Widget renames Tree.Widgets (W_Id);
   begin
      if not W.Visible then
         return;
      end if;

      declare
         Box : Rect renames W.Geometry.Border_Box;
         C   : CSS_Color renames W.Style.Background_Color;
      begin
         -- Draw background if non-transparent
         if C.A > 0.01 then
            Draw_Filled_Rect (
               Box.X, Box.Y, Box.Width, Box.Height,
               C.R, C.G, C.B, C.A * W.Style.Opacity);
         end if;

         -- Draw border if border-width > 0 (simplified: draw as thin rects)
         if W.Layout.Border_Width.Top > 0.0 then
            Draw_Filled_Rect (
               Box.X, Box.Y, Box.Width, W.Layout.Border_Width.Top,
               W.Style.Border_Color.R, W.Style.Border_Color.G,
               W.Style.Border_Color.B, W.Style.Border_Color.A);
         end if;

         -- Draw children (depth-first)
         for C_Idx in 1 .. W.Child_Count loop
            Render_Widget (Tree, W.Children (C_Idx));
         end loop;
      end;
   end Render_Widget;

   --  Render the entire widget tree.
   --  Clears the screen, then renders from root widget depth-first.
   --  Citation: OpenGL ES 2.0 §4.2 — glClear, glClearColor
   procedure Render_Tree (Tree : Widget_Tree) is
   begin
      -- Initialize renderer on first call
      if not G_Render_State.Initialized then
         Init_Renderer (1200.0, 800.0);
      end if;

      -- Clear the screen with background color (#0b0c0e)
      GL.Buffers.Set_Color_Clear_Value ((0.043, 0.047, 0.055, 1.0));
      GL.Buffers.Clear (GL.Buffers.Color => True);

      -- Render from root widget
      Render_Widget (Tree, Tree.Root_ID);
   end Render_Tree;

   -- =========================================================================
   -- INPUT HANDLING
   -- =========================================================================

   function Hit_Test
     (Tree : Widget_Tree;
      X    : Float;
      Y    : Float)
      return Widget_ID
   is
      --  Depth-first search, topmost (last drawn) wins.
      --  We iterate in reverse for correct overlap handling.
      Best_ID : Widget_ID := 0;
   begin
      for I in reverse 1 .. Tree.Widget_Count loop
         declare
            W : Widget renames Tree.Widgets (I);
         begin
            if W.Visible then
               declare
                  Box : Rect renames W.Geometry.Border_Box;
               begin
                  if X >= Box.X and then X <= Box.X + Box.Width
                    and then Y >= Box.Y and then Y <= Box.Y + Box.Height
                  then
                     Best_ID := W.ID;
                     exit;  -- Found the topmost widget
                  end if;
               end;
            end if;
         end;
      end loop;
      return Best_ID;
   end Hit_Test;

   function Process_Input
     (Tree  : in out Widget_Tree;
      Event : Input_Event)
      return Widget_ID
   is
      Target_ID : Widget_ID;
   begin
      case Event.Kind is
         when Event_Mouse_Move | Event_Mouse_Press | Event_Mouse_Release =>
            Target_ID := Hit_Test (Tree, Event.X, Event.Y);

            -- Update hover state
            for I in 1 .. Tree.Widget_Count loop
               Tree.Widgets (I).Hovered := (Tree.Widgets (I).ID = Target_ID);
            end loop;

            -- Handle click
            if Event.Kind = Event_Mouse_Press
              and then Target_ID > 0
            then
               Tree.Widgets (Target_ID).Active := True;

               -- Trigger On_Click callback if set
               if Tree.Widgets (Target_ID).On_Click > 0 then
                  Adelaide_Trace.Trace_Print (
                    Toolcall => "widget_tree:input",
                    Message => "Click on widget " &
                      Natural'Image (Integer (Target_ID)));
               end if;
            end if;

            if Event.Kind = Event_Mouse_Release then
               for I in 1 .. Tree.Widget_Count loop
                  Tree.Widgets (I).Active := False;
               end loop;
            end if;

            return Target_ID;

         when Event_Key_Press | Event_Key_Release =>
            -- Route to focused widget
            for I in 1 .. Tree.Widget_Count loop
               if Tree.Widgets (I).Focused then
                  -- Handle keyboard input for text fields
                  if Tree.Widgets (I).Kind = Widget_Input
                    and then Event.Kind = Event_Key_Press
                  then
                     if Event.Key_Char = ASCII.BS then
                        -- Backspace
                        if Length (Tree.Widgets (I).Input_Value) > 0 then
                           Delete (Tree.Widgets (I).Input_Value,
                             Length (Tree.Widgets (I).Input_Value), 1);
                        end if;
                     elsif Event.Key_Char /= ASCII.NUL
                       and then Event.Key_Char /= ASCII.CR
                       and then Event.Key_Char /= ASCII.LF
                     then
                        Append (Tree.Widgets (I).Input_Value, Event.Key_Char);
                     end if;
                  end if;
                  return Tree.Widgets (I).ID;
               end if;
            end loop;
            return 0;

         when Event_Scroll =>
            Target_ID := Hit_Test (Tree, Event.X, Event.Y);
            if Target_ID > 0
              and then Tree.Widgets (Target_ID).Kind = Widget_Scrollable
            then
               Tree.Widgets (Target_ID).Scroll_Y :=
                 Float'Max (0.0,
                   Float'Min (Tree.Widgets (Target_ID).Scroll_Max,
                     Tree.Widgets (Target_ID).Scroll_Y + Event.Scroll_Delta));
               return Target_ID;
            end if;
            return 0;

         when Event_Resize =>
            -- Update root widget dimensions
            Tree.Widgets (Tree.Root_ID).Geometry.Content_Box.Width := Event.Width;
            Tree.Widgets (Tree.Root_ID).Geometry.Content_Box.Height := Event.Height;
            -- Re-run layout
            Compute_Layout (Tree, Event.Width, Event.Height);
            return Tree.Root_ID;
      end case;
   end Process_Input;

   -- =========================================================================
   -- ANIMATION
   -- =========================================================================

   procedure Update_Animations
     (Tree       : in out Widget_Tree;
      Delta_Time : Float)
   is
   begin
      for I in 1 .. Tree.Widget_Count loop
         if Tree.Widgets (I).Animation.Is_Active then
            Tree.Widgets (I).Animation.Elapsed :=
              Tree.Widgets (I).Animation.Elapsed + Delta_Time;

            -- Apply animation effect
            case Tree.Widgets (I).Animation.Kind is
               when Anim_Pulse =>
                  -- Scale and opacity pulsing
                  declare
                     Phase : constant Float :=
                       Tree.Widgets (I).Animation.Elapsed /
                       Tree.Widgets (I).Animation.Duration;
                     Sin_Val : Float := 0.0;
                  begin
                     -- Simple sine approximation for pulse
                     -- sin(x) ≈ x * (pi - x) / (2.25 * (pi - x) + x * x) * 4
                     -- (Bhaskara I approximation, good enough for smooth pulse)
                     declare
                        Pi : constant Float := 3.14159265;
                        T  : Float := Phase - Float (Integer (Phase / 2.0)) * 2.0;
                     begin
                        if T > Pi then
                           T := 2.0 * Pi - T;
                        end if;
                        if T < Pi / 2.0 then
                           Sin_Val := T * (Pi - T) / (2.25 * (Pi - T) + T * T) * 4.0;
                        else
                           Sin_Val := (Pi - T) * T / (2.25 * T + (Pi - T) * (Pi - T)) * 4.0;
                        end if;
                     end;
                     Tree.Widgets (I).Style.Opacity := 0.8 + 0.2 * Sin_Val;
                  end;

               when Anim_Fade_In =>
                  -- Opacity 0 → 1 over Duration
                  declare
                     Progress : Float :=
                       Float'Min (1.0,
                         Tree.Widgets (I).Animation.Elapsed /
                         Tree.Widgets (I).Animation.Duration);
                  begin
                     Tree.Widgets (I).Style.Opacity := Progress;
                  end;

               when Anim_Float_Logo =>
                  -- Vertical floating (translate Y)
                  declare
                     Phase : constant Float :=
                       Tree.Widgets (I).Animation.Elapsed /
                       Tree.Widgets (I).Animation.Duration;
                     Pi    : constant Float := 3.14159265;
                     T     : Float := Phase - Float (Integer (Phase)) ;
                     Sin_Val : Float := 0.0;
                  begin
                     if T < 0.25 then
                        Sin_Val := T * 4.0;
                     elsif T < 0.75 then
                        Sin_Val := 1.0 - (T - 0.25) * 4.0;
                     else
                        Sin_Val := -1.0 + (T - 0.75) * 4.0;
                     end if;
                     -- Store offset in Scroll_Y (temporary hack)
                     Tree.Widgets (I).Scroll_Y := Sin_Val * 10.0;
                  end;

               when Anim_Twinkle =>
                  -- Opacity oscillation (star twinkling)
                  declare
                     Phase : constant Float :=
                       Tree.Widgets (I).Animation.Elapsed /
                       Tree.Widgets (I).Animation.Duration;
                     T     : Float := Phase - Float (Integer (Phase));
                  begin
                     Tree.Widgets (I).Style.Opacity := 0.1 + 0.9 *
                       (if T < 0.5 then T * 2.0 else 2.0 - T * 2.0);
                  end;

               when Anim_Shooting_Star | Anim_Spin_Gradient | Anim_None =>
                  null;
            end case;

            -- Reset elapsed if animation duration exceeded (loop)
            if Tree.Widgets (I).Animation.Elapsed >=
              Tree.Widgets (I).Animation.Duration
            then
               Tree.Widgets (I).Animation.Elapsed := 0.0;
            end if;
         end if;
      end loop;
   end Update_Animations;

   procedure Start_Animation
     (Tree     : in out Widget_Tree;
      ID       : Widget_ID;
      Anim     : Animation_Kind;
      Duration : Float)
   is
   begin
      if ID > 0 and then ID <= Tree.Widget_Count then
         Tree.Widgets (ID).Animation := (
            Kind       => Anim,
            Start_Time => 0.0,
            Duration   => Duration,
            Loop_Count => 0,
            Elapsed    => 0.0,
            Is_Active  => True
         );
      end if;
   end Start_Animation;

   -- =========================================================================
   -- TREE TRAVERSAL UTILITIES
   -- =========================================================================

   function Get_Children
     (Tree : Widget_Tree;
      ID   : Widget_ID)
      return Widget
   is
   begin
      if ID > 0 and then ID <= Tree.Widget_Count then
         return Tree.Widgets (ID);
      end if;
      return Tree.Widgets (1);  -- Return root as fallback
   end Get_Children;

   function Get_Parent
     (Tree : Widget_Tree;
      ID   : Widget_ID)
      return Widget
   is
   begin
      if ID > 0 and then ID <= Tree.Widget_Count then
         declare
            Parent_Id : constant Widget_ID := Tree.Widgets (ID).Parent;
         begin
            if Parent_Id > 0 and then Parent_Id <= Tree.Widget_Count then
               return Tree.Widgets (Parent_Id);
            end if;
         end;
      end if;
      return Tree.Widgets (1);  -- Return root as fallback
   end Get_Parent;

   function Is_Visible_In_Tree
     (Tree : Widget_Tree;
      ID   : Widget_ID)
      return Boolean
   is
      Current : Widget_ID := ID;
   begin
      while Current > 0 and then Current <= Tree.Widget_Count loop
         if not Tree.Widgets (Current).Visible then
            return False;
         end if;
         Current := Tree.Widgets (Current).Parent;
      end loop;
      return True;
   end Is_Visible_In_Tree;

end Zephyrine_Widget_Tree;
