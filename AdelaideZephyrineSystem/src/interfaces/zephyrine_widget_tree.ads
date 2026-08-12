pragma SPARK_Mode (Off);
-- ============================================================================
-- ZEPHYRINE_WIDGET_TREE — Widget system for the Zephyrine native renderer
-- ============================================================================
--
-- AXIOMS AND CITATIONS:
--   - Immediate Mode GUI (IMGUI) vs Retained Mode: This uses Retained Mode
--     (widget tree) because:
--     a) The UI is long-lived (entire app session), not transient overlays
--     b) CSS-style styling maps naturally to a tree of named elements
--     c) The existing HTML uses #id and .class selectors → widget IDs/classes
--   - Layout engine follows CSS Box Model (W3C CSS Box Model Level 3):
--     https://www.w3.org/TR/css-box-3/
--     Content → Padding → Border → Margin
--   - Flexbox layout (simplified) for sidebar/main split:
--     https://www.w3.org/TR/css-flexbox-1/
--     display: flex, flex-direction: column/row, justify-content, align-items
--   - Widget tree traversal: Depth-first, parent-to-child layout propagation
--
-- WIDGET TYPES (from style.css analysis):
--   1. Panel: #sidebar, #main-area, #chat-container — containers with layout
--   2. Button: .new-chat-btn, .nav-item, .msg-btn — clickable elements
--   3. Text: .greeting-title, .history-topic-label — text display
--   4. Input: #input-field — text input area
--   5. Scrollable: .history-section, #chat-container — scroll regions
--   6. Tab: Knowledge tabs, about tabs — tabbed panels
--   7. Image: .center-logo, avatar — image display
--   8. Background: #bg, .orb, .stars — animated background layers
--
-- ============================================================================

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Zephyrine_CSS_Parser;   use Zephyrine_CSS_Parser;

package Zephyrine_Widget_Tree is

   -- =========================================================================
   -- CONSTANTS
   -- =========================================================================

   Max_Children   : constant := 32;   -- Max children per widget
   Max_Widgets    : constant := 256;  -- Max total widgets in the tree
   Max_Animations : constant := 16;   -- Max concurrent animations

   -- =========================================================================
   -- TYPES — Widget identity and classification
   -- =========================================================================

   --  Widget_Kind: The type of UI element.
   --  Each kind has distinct rendering behavior and input handling.
   type Widget_Kind is
     (Widget_Panel,       -- Container: sidebar, main-area, sections
      Widget_Button,      -- Clickable: nav items, send button, action btns
      Widget_Text,        -- Text display: labels, paragraphs, titles
      Widget_Input,       -- Text input field: message composer
      Widget_Scrollable,  -- Scrollable container: history, chat messages
      Widget_Tab,         -- Tab system: knowledge tabs, about tabs
      Widget_Image,       -- Image display: logos, avatars, icons
      Widget_Background,  -- Animated background: orbs, stars, gradients
      Widget_Container);  -- Generic container: div, section, etc.

   --  Widget_ID: Unique identifier for a widget in the tree.
   --  Maps to HTML id attributes (#sidebar, #main-area, etc.)
   Max_Widget_ID_Length : constant := 64;
   type Widget_ID is new Positive range 1 .. Max_Widgets;

   --  Named array type for widget children (Ada requires named types in records)
   type Widget_ID_Array is array (1 .. Max_Children) of Widget_ID;

   --  Widget_Class: CSS class name(s) for a widget.
   --  Maps to HTML class attributes (.nav-item, .bubble, etc.)
   Max_Class_Length : constant := 128;
   type Widget_Class is record
      Raw : Unbounded_String;  -- Original class string (e.g. "nav-item active")
   end record;

   --  Widget_Tag: HTML tag name (div, button, input, etc.)
   Max_Tag_Length : constant := 32;
   type Widget_Tag is record
      Raw : Unbounded_String;
   end record;

   -- =========================================================================
   -- TYPES — Layout and geometry
   -- =========================================================================

   --  Rect: Axis-aligned rectangle in screen coordinates (pixels).
   --  Origin is top-left corner (OpenGL convention: Y-down after viewport flip).
   type Rect is record
      X      : Float := 0.0;  -- Left edge
      Y      : Float := 0.0;  -- Top edge
      Width  : Float := 0.0;  -- Width in pixels
      Height : Float := 0.0;  -- Height in pixels
   end record;

   --  Edge_Sizes: CSS box model edges (margin, border, padding).
   type Edge_Sizes is record
      Top    : Float := 0.0;
      Right  : Float := 0.0;
      Bottom : Float := 0.0;
      Left   : Float := 0.0;
   end record;

   --  Layout_Direction: Flexbox flex-direction.
   type Layout_Direction is
     (Dir_Column,    -- flex-direction: column (top to bottom)
      Dir_Row);     -- flex-direction: row (left to right)

   --  Justify_Content: Flexbox justify-content.
   type Justify_Kind is
     (Justify_Flex_Start,
      Justify_Flex_End,
      Justify_Center,
      Justify_Space_Between,
      Justify_Space_Around);

   --  Align_Kind: Flexbox align-items.
   type Align_Kind is
     (Align_Flex_Start,
      Align_Flex_End,
      Align_Center,
      Align_Stretch);

   --  Position_Kind: CSS position property.
   type Position_Kind is
     (Pos_Static,   -- position: static (default)
      Pos_Relative, -- position: relative
      Pos_Absolute, -- position: absolute
      Pos_Fixed);   -- position: fixed

   --  Overflow_Kind: CSS overflow property.
   type Overflow_Kind is
     (Overflow_Visible,
      Overflow_Hidden,
      Overflow_Scroll,
      Overflow_Auto);

   --  Widget_Layout: Layout parameters for a widget.
   type Widget_Layout is record
      Direction      : Layout_Direction := Dir_Column;
      Justify        : Justify_Kind := Justify_Flex_Start;
      Align          : Align_Kind := Align_Stretch;
      Flex_Grow      : Float := 0.0;
      Flex_Shrink    : Float := 1.0;
      Position       : Position_Kind := Pos_Static;
      Overflow       : Overflow_Kind := Overflow_Visible;
      Margin         : Edge_Sizes;
      Border_Width   : Edge_Sizes;
      Padding        : Edge_Sizes;
      Border_Radius  : Float := 0.0;
      Gap            : Float := 0.0;
   end record;

   --  Widget_Geometry: Resolved layout geometry (computed by layout engine).
   type Widget_Geometry is record
      Content_Box : Rect;   -- Content area (inside padding)
      Border_Box  : Rect;   -- Border area (inside margin)
      Margin_Box  : Rect;   -- Margin area (outermost)
   end record;

   -- =========================================================================
   -- TYPES — Visual properties (from CSS)
   -- =========================================================================

   --  Widget_Style: Visual style properties applied to a widget.
   type Widget_Style is record
      Background_Color : CSS_Color := (0.055, 0.047, 0.055, 1.0);  -- #0b0c0e
      Text_Color       : CSS_Color := (0.89, 0.89, 0.89, 1.0);     -- #e3e3e3
      Border_Color     : CSS_Color := (0.0, 0.0, 0.0, 0.0);        -- transparent
      Opacity          : Float := 1.0;
      Border_Radius    : Float := 0.0;
      Box_Shadow       : CSS_Color := (0.0, 0.0, 0.0, 0.0);        -- shadow color
      Box_Shadow_Blur  : Float := 0.0;
      Font_Size        : Float := 16.0;  -- in pixels
      Font_Weight      : Natural := 400;  -- 400=normal, 700=bold
      Line_Height      : Float := 1.5;
      Letter_Spacing   : Float := 0.0;
   end record;

   -- =========================================================================
   -- TYPES — Animation state
   -- =========================================================================

   --  Animation_Kind: Types of CSS animations used in style.css.
   type Animation_Kind is
     (Anim_None,
      Anim_Pulse,        -- pulseOrb: scale/opacity pulsing
      Anim_Twinkle,      -- twinkle: star twinkling
      Anim_Shooting_Star, -- shootingStar: diagonal slide
      Anim_Fade_In,      -- fadeIn: opacity 0 → 1
      Anim_Spin_Gradient, -- spinGradient: background-position shift
      Anim_Float_Logo);  -- floatLogo: vertical float

   --  Animation_State: Current state of an animation on a widget.
   type Animation_State is record
      Kind        : Animation_Kind := Anim_None;
      Start_Time  : Float := 0.0;   -- Seconds since app start
      Duration    : Float := 1.0;   -- Total animation duration
      Loop_Count  : Natural := 0;   -- 0 = infinite
      Elapsed     : Float := 0.0;   -- Current time in the animation
      Is_Active   : Boolean := False;
   end record;

   -- =========================================================================
   -- TYPES — Widget record (tree node)
   -- =========================================================================

   --  Widget: A single node in the widget tree.
   --  Each widget represents one UI element (div, button, text, etc.)
   type Widget is record
      --  Identity
      ID        : Widget_ID;
      Kind      : Widget_Kind;
      Tag       : Widget_Tag;
      Class     : Widget_Class;

      --  Tree structure
      Parent    : Widget_ID := 0;  -- 0 = root (no parent)
       Children  : Widget_ID_Array;
      Child_Count : Natural := 0;

      --  Layout
      Layout    : Widget_Layout;
      Geometry  : Widget_Geometry;
      Style     : Widget_Style;

      --  Animation
      Animation : Animation_State;

      --  State
      Visible   : Boolean := True;
      Focused   : Boolean := False;  -- For input fields
      Hovered   : Boolean := False;  -- For hover effects
      Active    : Boolean := False;  -- For pressed/clicked state

      --  Content (for text/image widgets)
      Text_Content   : Unbounded_String;
      Image_Path     : Unbounded_String;

      --  Input state (for Widget_Input)
      Input_Value    : Unbounded_String;
      Input_Cursor   : Natural := 0;

      --  Scroll state (for Widget_Scrollable)
      Scroll_Y       : Float := 0.0;
      Scroll_Max     : Float := 0.0;
      Viewport_Height : Float := 0.0;

      --  Tab state (for Widget_Tab)
      Active_Tab     : Natural := 0;

      --  Callback (for buttons)
      On_Click       : Natural := 0;  -- Callback ID (0 = none)
    end record;

    --  Named array type for widget tree storage
    type Widget_Array is array (1 .. Max_Widgets) of Widget;

    -- =========================================================================
    -- TYPES — The complete widget tree
    -- =========================================================================

    --  Widget_Tree: The entire UI widget hierarchy.
    type Widget_Tree is record
      Widgets    : Widget_Array;
      Widget_Count : Natural := 0;
      Root_ID    : Widget_ID := 1;  -- The root widget
   end record;

   -- =========================================================================
   -- TYPES — Input events
   -- =========================================================================

   --  Input_Event: Types of user input.
   type Input_Event_Kind is
     (Event_Mouse_Move,
      Event_Mouse_Press,
      Event_Mouse_Release,
      Event_Key_Press,
      Event_Key_Release,
      Event_Scroll,
      Event_Resize);

   --  Mouse_Button: Which mouse button was pressed.
   type Mouse_Button is
     (Button_None,
      Button_Left,
      Button_Right,
      Button_Middle);

   --  Input_Event: A single input event from the platform.
   type Input_Event is record
      Kind       : Input_Event_Kind;
      X          : Float := 0.0;  -- Mouse X position
      Y          : Float := 0.0;  -- Mouse Y position
      Button     : Mouse_Button := Button_None;
      Key_Code   : Natural := 0;  -- Virtual key code
      Key_Char   : Character := ASCII.NUL;
      Scroll_Delta : Float := 0.0;
      Width      : Float := 0.0;  -- Window dimensions (for resize)
      Height     : Float := 0.0;
   end record;

   -- =========================================================================
   -- PROCEDURES — Tree construction
   -- =========================================================================

   --  Init_Tree: Initialize an empty widget tree with the root widget.
   procedure Init_Tree (Tree : in out Widget_Tree);

   --  Add_Widget: Create a new widget and add it to the tree.
   --  Returns the Widget_ID of the newly created widget.
   function Add_Widget
     (Tree      : in out Widget_Tree;
      Kind      : Widget_Kind;
      Tag       : String;
      Class     : String;
      Parent_ID : Widget_ID := 0)
      return Widget_ID;

   --  Remove_Widget: Remove a widget and all its descendants.
   procedure Remove_Widget
     (Tree : in out Widget_Tree;
      ID   : Widget_ID);

   --  Find_Widget_By_ID: Find a widget by its HTML-like ID string.
   --  Returns 0 if not found.
   function Find_Widget_By_ID
     (Tree       : Widget_Tree;
      Search_ID  : String)
      return Widget_ID;

   --  Find_Widget_By_Class: Find the first widget matching a CSS class.
   function Find_Widget_By_Class
     (Tree         : Widget_Tree;
      Search_Class : String)
      return Widget_ID;

   -- =========================================================================
   -- PROCEDURES — Layout engine
   -- =========================================================================

   --  Apply_CSS_Stylesheet: Walk the widget tree and apply CSS properties
   --  from the parsed stylesheet. This resolves background colors, font sizes,
   --  border radii, padding, margins, etc.
   procedure Apply_CSS_Stylesheet
     (Tree       : in out Widget_Tree;
      Stylesheet : CSS_Stylesheet);

   --  Compute_Layout: Run the layout engine to compute widget geometry.
   --  Starting from the root, recursively computes content/border/margin
   --  boxes for every visible widget.
   procedure Compute_Layout
     (Tree       : in out Widget_Tree;
      Root_Width : Float;
      Root_Height: Float);

   -- =========================================================================
   -- PROCEDURES — Rendering
   -- =========================================================================

   --  Render_Tree: Traverse the widget tree and issue OpenGL ES 2.0 draw
   --  commands for each visible widget. Performs depth-first traversal.
   --
   --  Rendering pipeline per widget:
   --    1. Set scissor rect (clip to parent's content box)
   --    2. Draw background (quad with background color)
   --    3. Draw border (if border-width > 0)
   --    4. Draw text (if text content is non-empty)
   --    5. Draw children recursively
   --
   --  This procedure calls the GLESv2_Binding procedures directly.
   procedure Render_Tree (Tree : Widget_Tree);

   -- =========================================================================
   -- PROCEDURES — Input handling
   -- =========================================================================

   --  Process_Input: Route an input event to the appropriate widget.
   --  Returns the Widget_ID that consumed the event (0 if none).
   function Process_Input
     (Tree  : in out Widget_Tree;
      Event : Input_Event)
      return Widget_ID;

   --  Hit_Test: Find which widget is under the given screen coordinates.
   --  Uses the widget geometry (border box) for hit detection.
   function Hit_Test
     (Tree : Widget_Tree;
      X    : Float;
      Y    : Float)
      return Widget_ID;

   -- =========================================================================
   -- PROCEDURES — Animation
   -- =========================================================================

   --  Update_Animations: Advance all active animations by Delta_Time seconds.
   procedure Update_Animations
     (Tree       : in out Widget_Tree;
      Delta_Time : Float);

   --  Start_Animation: Begin an animation on a widget.
   procedure Start_Animation
     (Tree     : in out Widget_Tree;
      ID       : Widget_ID;
      Anim     : Animation_Kind;
      Duration : Float);

   -- =========================================================================
   -- PROCEDURES — Tree traversal utilities
   -- =========================================================================

   --  For_Each_Widget: Iterate over all widgets (depth-first).
   --  The callback receives each Widget_ID. (Simplified: no actual callback
   --  type in Ada — callers use a simple loop over Widget_Count.)

   --  Get_Children: Return the children of a widget.
   function Get_Children
     (Tree : Widget_Tree;
      ID   : Widget_ID)
      return Widget;

   --  Get_Parent: Return the parent of a widget.
   function Get_Parent
     (Tree : Widget_Tree;
      ID   : Widget_ID)
      return Widget;

   --  Is_Visible_In_Tree: Check if a widget is visible (all ancestors visible).
   function Is_Visible_In_Tree
     (Tree : Widget_Tree;
      ID   : Widget_ID)
      return Boolean;

end Zephyrine_Widget_Tree;
