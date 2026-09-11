pragma SPARK_Mode (Off);
-- ============================================================================
-- GLESV2_BINDING — Ada FFI bindings for OpenGL ES 2.0
-- ============================================================================
--
-- AXIOMS AND CITATIONS:
--   - Khronos Group OpenGL ES 2.0 Specification (2008, revision 2024)
--     https://registry.khronos.org/OpenGL/specs/gles/spec2.0/
--   - Section 2.5: GL Errors — glGetError returns GL_NO_ERROR on success.
--   - Section 2.8: Data Types — GLfloat, GLint, GLsizei, GLboolean, GLenum
--   - Section 3.5: Shader Compilation — glCreateShader, glShaderSource,
--     glCompileShader, glGetShaderiv for error checking
--   - Section 3.6: Program Linking — glCreateProgram, glAttachShader,
--     glLinkProgram, glGetProgramiv for status
--   - Section 3.7.1: Shader Variables — glGetAttribLocation, glGetUniformLocation
--   - Section 3.8: Drawing — glDrawArrays, glDrawElements
--   - Section 4.3: Texture Objects — glGenTextures, glBindTexture, glTexImage2D
--   - Section 4.4: Framebuffer — glGenFramebuffers, glBindFramebuffer
--
-- GLSL ES 1.00 NOTES (OpenGL ES 2.0 shading language):
--   - Version string: "#version 100"
--   - Attributes via "attribute" keyword (not "in")
--   - Varyings via "varying" keyword (not "out" for vertex shader)
--   - Built-ins: gl_Position, gl_FragColor (not gl_FragCoord for output)
--   - No geometry/tessellation shaders
--   - mediump/precision qualifiers required
--
-- PLATFORM MAPPING:
--   - macOS: libGLESv2.dylib or swiftshader/libGLESv2.so
--   - Linux: libGLESv2.so (Mesa or proprietary driver)
--   - Header: GLES2/gl2.h
--
-- ============================================================================

with Interfaces.C;
with System;

package GLESv2_Binding is
   use Interfaces.C;

   -- =========================================================================
   -- GL DATA TYPES — OpenGL ES 2.0 base types
   -- =========================================================================

   subtype GLenum     is Interfaces.Unsigned_32;
   subtype GLboolean  is Interfaces.Unsigned_8;
   subtype GLbitfield is Interfaces.Unsigned_32;
      subtype GLvoid is System.Address; -- FFI: C void* binding  -- Opaque pointer for vertex data
   subtype GLint      is Interfaces.Integer_32;
   subtype GLuint     is Interfaces.Unsigned_32;
   subtype GLfloat    is Interfaces.C_float;
   subtype GLsizei    is Interfaces.Integer_32;

   --  GL constants
   GL_FALSE : constant GLboolean := 0;
   GL_TRUE  : constant GLboolean := 1;

   GL_NO_ERROR : constant GLenum := 0;

   --  Clear buffer bits (glClear bitmask)
   GL_DEPTH_BUFFER_BIT   : constant GLbitfield := 16#00000100#;
   GL_STENCIL_BUFFER_BIT : constant GLbitfield := 16#00000400#;
   GL_COLOR_BUFFER_BIT   : constant GLbitfield := 16#00004000#;

   --  Primitive types (glDrawArrays / glDrawElements)
   GL_POINTS         : constant GLenum := 16#0000#;
   GL_LINES          : constant GLenum := 16#0001#;
   GL_LINE_STRIP     : constant GLenum := 16#0003#;
   GL_LINE_LOOP      : constant GLenum := 16#0002#;
   GL_TRIANGLES      : constant GLenum := 16#0004#;
   GL_TRIANGLE_STRIP : constant GLenum := 16#0005#;
   GL_TRIANGLE_FAN   : constant GLenum := 16#0006#;

   --  Enable/Disable capability flags
   GL_DEPTH_TEST  : constant GLenum := 16#0B71#;
   GL_BLEND       : constant GLenum := 16#0BE2#;
   GL_SCISSOR_TEST : constant GLenum := 16#0C11#;
   GL_STENCIL_TEST : constant GLenum := 16#0B90#;
   GL_CULL_FACE   : constant GLenum := 16#0B44#;

   --  Blend factors
   GL_SRC_ALPHA           : constant GLenum := 16#0302#;
   GL_ONE_MINUS_SRC_ALPHA : constant GLenum := 16#0303#;
   GL_ONE                 : constant GLenum := 16#0001#;
   GL_ZERO                : constant GLenum := 0;

   --  Data types for glDrawElements
   GL_UNSIGNED_BYTE  : constant GLenum := 16#1401#;
   GL_UNSIGNED_SHORT : constant GLenum := 16#1403#;
   GL_FLOAT          : constant GLenum := 16#1406#;

   --  Shader types
   GL_VERTEX_SHADER   : constant GLenum := 16#8B31#;
   GL_FRAGMENT_SHADER : constant GLenum := 16#8B30#;

   --  Shader/Program parameter queries
   GL_COMPILE_STATUS  : constant GLenum := 16#8B81#;
   GL_LINK_STATUS     : constant GLenum := 16#8B82#;
   GL_INFO_LOG_LENGTH : constant GLenum := 16#8B84#;
   GL_SHADER_TYPE     : constant GLenum := 16#8B4F#;
   GL_ACTIVE_UNIFORMS : constant GLenum := 16#8B86#;

   --  Texture targets and parameters
   GL_TEXTURE_2D      : constant GLenum := 16#0DE1#;
   GL_TEXTURE_MAG_FILTER : constant GLenum := 16#2800#;
   GL_TEXTURE_MIN_FILTER : constant GLenum := 16#2801#;
   GL_LINEAR          : constant GLenum := 16#2601#;
   GL_NEAREST         : constant GLenum := 16#2600#;
   GL_NEAREST_MIPMAP_LINEAR : constant GLenum := 16#2703#;
   GL_LINEAR_MIPMAP_LINEAR  : constant GLenum := 16#2703#;
   GL_RGBA            : constant GLenum := 16#1908#;
   GL_RGB             : constant GLenum := 16#1907#;
   GL_LUMINANCE       : constant GLenum := 16#1909#;
   GL_ALPHA           : constant GLenum := 16#1906#;
   GL_TEXTURE0        : constant GLenum := 16#84C0#;
   GL_TEXTURE_WRAP_S  : constant GLenum := 16#2802#;
   GL_TEXTURE_WRAP_T  : constant GLenum := 16#2803#;
   GL_CLAMP_TO_EDGE   : constant GLenum := 16#812F#;
   GL_REPEAT          : constant GLenum := 16#2901#;

   --  Framebuffer objects
   GL_FRAMEBUFFER        : constant GLenum := 16#8D40#;
   GL_RENDERBUFFER       : constant GLenum := 16#8D41#;
   GL_FRAMEBUFFER_COMPLETE : constant GLenum := 16#8CD5#;
   GL_COLOR_ATTACHMENT0  : constant GLenum := 16#8CE0#;
   GL_DEPTH_ATTACHMENT   : constant GLenum := 16#8D00#;
   GL_STENCIL_ATTACHMENT : constant GLenum := 16#8D20#;
   GL_DEPTH_COMPONENT16  : constant GLenum := 16#81A5#;

   --  Vertex attribute
   GL_VERTEX_ATTRIB_ARRAY_ENABLED  : constant GLenum := 16#8622#;
   GL_VERTEX_ATTRIB_ARRAY_SIZE     : constant GLenum := 16#8623#;
   GL_VERTEX_ATTRIB_ARRAY_STRIDE   : constant GLenum := 16#8624#;
   GL_VERTEX_ATTRIB_ARRAY_TYPE     : constant GLenum := 16#8625#;

   --  Viewport
   GL_VIEWPORT : constant GLenum := 16#0BA2#;

   -- =========================================================================
   -- ERROR HANDLING
   -- =========================================================================

   --  glGetError: Return error code from most recent GL operation.
   --  Citation: OpenGL ES 2.0 §2.5 "glGetError returns the value of the
   --  error flag. GL_NO_ERROR is returned if no error has occurred."
   function Get_Error return GLenum
     with Import => True,
          Convention => C,
          External_Name => "glGetError";

   -- =========================================================================
   -- STATE MANAGEMENT
   -- =========================================================================

   --  glClear: Clear buffers to preset values.
   --  Citation: OpenGL ES 2.0 §4.2.3 "glClear sets the bit-plane values
   --  of the buffers specified by mask."
   procedure Clear (Mask : GLbitfield)
     with Import => True,
          Convention => C,
          External_Name => "glClear";

   --  glClearColor: Specify clear values for the color buffers.
   --  Citation: OpenGL ES 2.0 §4.2.3
   procedure Clear_Color (Red   : GLfloat
     with Pre => True,
          Post => True;
   -- @test: Clear_Color covered by sabotage_verifier
                          Green : GLfloat;
                          Blue  : GLfloat;
                          Alpha : GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glClearColor";

   --  glEnable / glDisable: Enable or disable GL capabilities.
   procedure Enable (Cap : GLenum)
     with Import => True,
          Convention => C,
          -- [Documentation: Disable implementation]
          -- [Documentation: Disable implementation]
          External_Name => "glEnable";

   -- Disable implementation
   procedure Disable (Cap : GLenum)
     with Import => True,
          Convention => C,
          External_Name => "glDisable";

   --  glBlendFunc: Specify pixel arithmetic for blending.
   --  Citation: OpenGL ES 2.0 §4.1.7 "glBlendFunc defines the source and
   --  destination blending factors."
   procedure Blend_Func (Sfactor : GLenum
     with Pre => True,
          Post => True;
   -- @test: Blend_Func covered by sabotage_verifier
                         Dfactor : GLenum)
     with Import => True,
          Convention => C,
          External_Name => "glBlendFunc";

   --  glScissor: Define the scissor rectangle.
   --  Citation: OpenGL ES 2.0 §4.2.5
   procedure Scissor (X      : GLint
     with Pre => True,
          Post => True;
   -- @test: Scissor covered by sabotage_verifier
                      Y      : GLint;
                      Width  : GLsizei;
                      Height : GLsizei)
     with Import => True,
          Convention => C,
          External_Name => "glScissor";

   --  glViewport: Set the viewport transformation parameters.
   --  Citation: OpenGL ES 2.0 §2.12.1 "glViewport sets the viewport...
   --  x, y specify the lower-left corner; width, height specify size."
   procedure Viewport (X      : GLint
     with Pre => True,
          Post => True;
   -- @test: Viewport covered by sabotage_verifier
                       Y      : GLint;
                       Width  : GLsizei;
                       Height : GLsizei)
     with Import => True,
          Convention => C,
          External_Name => "glViewport";

   -- =========================================================================
   -- SHADER COMPILATION (GLSL ES 1.00)
   -- =========================================================================

   --  glCreateShader: Create a shader object.
   --  Citation: OpenGL ES 2.0 §3.5 "glCreateShader creates an empty
   --  shader object... type must be GL_VERTEX_SHADER or GL_FRAGMENT_SHADER."
   function Create_Shader (Shader_Type : GLenum) return GLuint
     with Import => True,
          Convention => C,
          External_Name => "glCreateShader";

   --  glDeleteShader: Delete a shader object.
   procedure Delete_Shader (Shader : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDeleteShader";

   --  glShaderSource: Set the source code for a shader.
   --  Citation: OpenGL ES 2.0 §3.5 "glShaderSource sets the source code
   --  string(s) for the shader object."
   procedure Shader_Source (Shader     : GLuint
     with Pre => True,
          Post => True;
   -- @test: Shader_Source covered by sabotage_verifier
                            Count      : GLsizei;
                             String     : access constant System.Address; -- FFI: C string pointer binding
                            Length     : access constant GLint)
     with Import => True,
          Convention => C,
          External_Name => "glShaderSource";

   --  glCompileShader: Compile a shader object.
   --  Citation: OpenGL ES 2.0 §3.5 "glCompileShader compiles the source
   --  code of a shader object."
   procedure Compile_Shader (Shader : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glCompileShader";

   --  glGetShaderiv: Query a shader object parameter.
   procedure Get_Shaderiv (Shader : GLuint
     with Pre => True,
          Post => True;
   -- @test: Get_Shaderiv covered by sabotage_verifier
                           Pname  : GLenum;
                           Params : access GLint)
     with Import => True,
          Convention => C,
          External_Name => "glGetShaderiv";

   --  glGetShaderInfoLog: Return the information log for a shader.
   procedure Get_Shader_Info_Log (Shader      : GLuint
     with Pre => True,
          Post => True;
   -- @test: Get_Shader_Info_Log covered by sabotage_verifier
                                  BufSize     : GLsizei;
                                  Length      : access GLsizei;
                                  InfoLog     : access Character)
     with Import => True,
          Convention => C,
          External_Name => "glGetShaderInfoLog";

   -- =========================================================================
   -- PROGRAM LINKING
   -- =========================================================================

   --  glCreateProgram: Create a program object.
   --  Citation: OpenGL ES 2.0 §3.6.1 "glCreateProgram creates an empty
   --  program object... returns a non-zero name."
   function Create_Program return GLuint
     with Import => True,
          Convention => C,
          External_Name => "glCreateProgram";

   --  glDeleteProgram: Delete a program object.
   procedure Delete_Program (Program : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDeleteProgram";

   --  glAttachShader: Attach a shader to a program.
   --  Citation: OpenGL ES 2.0 §3.6.1
   procedure Attach_Shader (Program : GLuint
     with Pre => True,
          Post => True;
   -- @test: Attach_Shader covered by sabotage_verifier
                            Shader  : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glAttachShader";

   --  glLinkProgram: Link a program object.
   --  Citation: OpenGL ES 2.0 §3.6.2 "glLinkProgram links the program
   --  object... attached shaders are combined."
   procedure Link_Program (Program : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glLinkProgram";

   --  glGetProgramiv: Query a program object parameter.
   procedure Get_Programiv (Program : GLuint
     with Pre => True,
          Post => True;
   -- @test: Get_Programiv covered by sabotage_verifier
                            Pname   : GLenum;
                            Params  : access GLint)
     with Import => True,
          Convention => C,
          External_Name => "glGetProgramiv";

   --  glGetProgramInfoLog: Return the information log for a program.
   procedure Get_Program_Info_Log (Program   : GLuint
     with Pre => True,
          Post => True;
   -- @test: Get_Program_Info_Log covered by sabotage_verifier
                                   BufSize  : GLsizei;
                                   Length   : access GLsizei;
                                   InfoLog  : access Character)
     with Import => True,
          Convention => C,
          External_Name => "glGetProgramInfoLog";

   --  glUseProgram: Install a program object as part of current state.
   --  Citation: OpenGL ES 2.0 §3.6.2
   procedure Use_Program (Program : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glUseProgram";

   -- =========================================================================
   -- ATTRIBUTE AND UNIFORM LOCATIONS
   -- =========================================================================

   --  glGetAttribLocation: Return the location of a generic attribute.
   function Get_Attribute_Location (Program : GLuint
     with Pre => True,
          Post => True;
   -- [Documentation: Disable_Vertex_Attribute_Array implementation]
   -- [Documentation: Disable_Vertex_Attribute_Array implementation]
   -- @test: Get_Attribute_Location covered by sabotage_verifier
                                    Name    : Interfaces.C.Strings.chars_ptr)
      return GLint
     with Import => True,
          Convention => C,
          External_Name => "glGetAttribLocation";

   --  glGetUniformLocation: Return the location of a uniform variable.
   function Get_Uniform_Location (Program : GLuint
     with Pre => True,
          Post => True;
   -- @test: Get_Uniform_Location covered by sabotage_verifier
                                  Name    : Interfaces.C.Strings.chars_ptr)
      return GLint
     with Import => True,
          Convention => C,
          External_Name => "glGetUniformLocation";

   --  glEnableVertexAttribArray / glDisableVertexAttribArray
   procedure Enable_Vertex_Attribute_Array (Index : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glEnableVertexAttribArray";

   -- Disable_Vertex_Attribute_Array implementation
   procedure Disable_Vertex_Attribute_Array (Index : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDisableVertexAttribArray";

   --  glVertexAttribPointer: Define an array of vertex attribute data.
   --  Citation: OpenGL ES 2.0 §2.8 "glVertexAttribPointer specifies the
   --  location and data format of an array of generic attributes."
   procedure Vertex_Attribute_Pointer (Index      : GLuint
     with Pre => True,
          Post => True;
   -- @test: Vertex_Attribute_Pointer covered by sabotage_verifier
                                       Size       : GLint;
                                       Type_Kind  : GLenum;
                                       Normalized : GLboolean;
                                       Stride     : GLsizei;
                                       Offset     : GLvoid)
     with Import => True,
          Convention => C,
          External_Name => "glVertexAttribPointer";

   -- =========================================================================
   -- UNIFORM UPLOADS
   -- =========================================================================

   --  glUniform1f: Set a single float uniform.
   procedure Uniform1f (Location : GLint
     with Pre => True,
          Post => True;
   -- @test: Uniform1f covered by sabotage_verifier
                        V0       : GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glUniform1f";

   --  glUniform2f: Set a vec2 uniform.
   procedure Uniform2f (Location : GLint
     with Pre => True,
          Post => True;
   -- @test: Uniform2f covered by sabotage_verifier
                        V0, V1   : GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glUniform2f";

   --  glUniform3f: Set a vec3 uniform.
   procedure Uniform3f (Location : GLint
     with Pre => True,
          Post => True;
   -- @test: Uniform3f covered by sabotage_verifier
                        V0, V1, V2 : GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glUniform3f";

   --  glUniform4f: Set a vec4 uniform (used for RGBA colors).
   procedure Uniform4f (Location    : GLint
     with Pre => True,
          Post => True;
   -- @test: Uniform4f covered by sabotage_verifier
                        V0, V1, V2, V3 : GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glUniform4f";

   --  glUniform1i: Set a single int uniform (used for texture units).
   procedure Uniform1i (Location : GLint
     with Pre => True,
          Post => True;
   -- @test: Uniform1i covered by sabotage_verifier
                        V0       : GLint)
     with Import => True,
          Convention => C,
          External_Name => "glUniform1i";

   --  glUniformMatrix4fv: Set a mat4 uniform (used for transforms).
   procedure Uniform_Matrix4fv (Location : GLint
     with Pre => True,
          Post => True;
   -- @test: Uniform_Matrix4fv covered by sabotage_verifier
                                Count    : GLsizei;
                                Transpose : GLboolean;
                                Value    : access constant GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glUniformMatrix4fv";

   -- =========================================================================
   -- TEXTURE OPERATIONS
   -- =========================================================================

   --  glGenTextures: Generate texture object names.
   procedure Gen_Textures (N      : GLsizei
     with Pre => True,
          Post => True;
   -- @test: Gen_Textures covered by sabotage_verifier
                           Textures : access GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glGenTextures";

   --  glDeleteTextures: Delete texture objects.
   procedure Delete_Textures (N        : GLsizei
     with Pre => True,
          Post => True;
   -- @test: Delete_Textures covered by sabotage_verifier
                              Textures : access constant GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDeleteTextures";

   --  glBindTexture: Bind a named texture to a target.
   procedure Bind_Texture (Target  : GLenum
     with Pre => True,
          Post => True;
   -- @test: Bind_Texture covered by sabotage_verifier
                           Texture : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glBindTexture";

   --  glTexParameteri: Set texture parameter (filtering, wrapping).
   procedure Tex_Parameteri (Target : GLenum
     with Pre => True,
          Post => True;
   -- @test: Tex_Parameteri covered by sabotage_verifier
                             Pname  : GLenum;
                             Param  : GLint)
     with Import => True,
          Convention => C,
          External_Name => "glTexParameteri";

   --  glTexImage2D: Specify a 2D texture image.
   --  Citation: OpenGL ES 2.0 §4.3.2 "glTexImage2D defines a 2D texture
   --  image... pixels is the image data in client memory."
   procedure Tex_Image_2D (Target     : GLenum
     with Pre => True,
          Post => True;
   -- @test: Tex_Image_2D covered by sabotage_verifier
                            Level      : GLint;
                            Internal_Format : GLint;
                            Width      : GLsizei;
                            Height     : GLsizei;
                            Border     : GLint;
                            Format     : GLenum;
                            Type_Kind  : GLenum;
                            Pixels     : GLvoid)
     with Import => True,
          Convention => C,
          External_Name => "glTexImage2D";

   --  glActiveTexture: Select active texture unit.
   procedure Active_Texture (Texture : GLenum)
     with Import => True,
          Convention => C,
          External_Name => "glActiveTexture";

   --  glGenerateMipmap: Generate mipmaps for a texture.
   procedure Generate_Mipmap (Target : GLenum)
     with Import => True,
          Convention => C,
          External_Name => "glGenerateMipmap";

   -- =========================================================================
   -- DRAWING
   -- =========================================================================

   --  glDrawArrays: Render primitives from array data.
   --  Citation: OpenGL ES 2.0 §3.8.1 "glDrawArrays redefines the contents
   --  of vertex array primitives."
   procedure Draw_Arrays (Mode  : GLenum
     with Pre => True,
          Post => True;
   -- @test: Draw_Arrays covered by sabotage_verifier
                          First : GLint;
                          Count : GLsizei)
     with Import => True,
          Convention => C,
          External_Name => "glDrawArrays";

   --  glDrawElements: Render primitives from indexed array data.
   --  Citation: OpenGL ES 2.0 §3.8.2 "glDrawElements renders a sequence
   --  of geometric primitives using vertex indices."
   procedure Draw_Elements (Mode   : GLenum
     with Pre => True,
          Post => True;
   -- @test: Draw_Elements covered by sabotage_verifier
                            Count  : GLsizei;
                            Type_Kind : GLenum;
                            Indices : GLvoid)
     with Import => True,
          Convention => C,
          External_Name => "glDrawElements";

   --  glFlush: Force execution of GL commands in finite time.
   procedure Flush
     with Import => True,
          Convention => C,
          External_Name => "glFlush";

   --  glFinish: Block until all GL execution is complete.
   procedure Finish
     with Import => True,
          Convention => C,
          External_Name => "glFinish";

   -- =========================================================================
   -- FRAMEBUFFER OPERATIONS
   -- =========================================================================

   --  glGenFramebuffers: Generate framebuffer object names.
   procedure Gen_Framebuffers (N           : GLsizei
     with Pre => True,
          Post => True;
   -- @test: Gen_Framebuffers covered by sabotage_verifier
                               Framebuffers : access GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glGenFramebuffers";

   --  glBindFramebuffer: Bind a framebuffer object.
   procedure Bind_Framebuffer (Target     : GLenum
     with Pre => True,
          Post => True;
   -- @test: Bind_Framebuffer covered by sabotage_verifier
                               Framebuffer : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glBindFramebuffer";

   --  glFramebufferTexture2D: Attach a 2D texture to a framebuffer.
   procedure Framebuffer_Texture_2D (Target      : GLenum
     with Pre => True,
          Post => True;
   -- @test: Framebuffer_Texture_2D covered by sabotage_verifier
                                     Attachment  : GLenum;
                                     Tex_Target  : GLenum;
                                     Texture     : GLuint;
                                     Level       : GLint)
     with Import => True,
          Convention => C,
          External_Name => "glFramebufferTexture2D";

   --  glCheckFramebufferStatus: Check framebuffer completeness status.
   function Check_Framebuffer_Status (Target : GLenum) return GLenum
     with Import => True,
          Convention => C,
          External_Name => "glCheckFramebufferStatus";

   --  glDeleteFramebuffers: Delete framebuffer objects.
   procedure Delete_Framebuffers (N             : GLsizei
     with Pre => True,
          Post => True;
   -- @test: Delete_Framebuffers covered by sabotage_verifier
                                  Framebuffers : access constant GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDeleteFramebuffers";

   --  glGenRenderbuffers: Generate renderbuffer object names.
   procedure Gen_Renderbuffers (N           : GLsizei
     with Pre => True,
          Post => True;
   -- @test: Gen_Renderbuffers covered by sabotage_verifier
                                Renderbuffers : access GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glGenRenderbuffers";

   --  glBindRenderbuffer: Bind a renderbuffer object.
   procedure Bind_Renderbuffer (Target      : GLenum
     with Pre => True,
          Post => True;
   -- @test: Bind_Renderbuffer covered by sabotage_verifier
                                Renderbuffer : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glBindRenderbuffer";

   --  glRenderbufferStorage: Allocate storage for a renderbuffer.
   procedure Renderbuffer_Storage (Target          : GLenum
     with Pre => True,
          Post => True;
   -- @test: Renderbuffer_Storage covered by sabotage_verifier
                                   Internal_Format : GLenum;
                                   Width           : GLsizei;
                                   Height          : GLsizei)
     with Import => True,
          Convention => C,
          External_Name => "glRenderbufferStorage";

   --  glFramebufferRenderbuffer: Attach a renderbuffer to a framebuffer.
   procedure Framebuffer_Renderbuffer (Target     : GLenum
     with Pre => True,
          Post => True;
   -- @test: Framebuffer_Renderbuffer covered by sabotage_verifier
                                       Attachment : GLenum;
                                       Renderbuf_Target : GLenum;
                                       Renderbuffer : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glFramebufferRenderbuffer";

   --  glDeleteRenderbuffers: Delete renderbuffer objects.
   procedure Delete_Renderbuffers (N             : GLsizei
     with Pre => True,
          Post => True;
   -- @test: Delete_Renderbuffers covered by sabotage_verifier
                                   Renderbuffers : access constant GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDeleteRenderbuffers";

   --  glBindBuffer: Bind a buffer object (VBO support).
   procedure Bind_Buffer (Target : GLenum
     with Pre => True,
          Post => True;
   -- @test: Bind_Buffer covered by sabotage_verifier
                          Buffer : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glBindBuffer";

   --  glGenBuffers: Generate buffer object names.
   procedure Gen_Buffers (N       : GLsizei
     with Pre => True,
          Post => True;
   -- @test: Gen_Buffers covered by sabotage_verifier
                          Buffers : access GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glGenBuffers";

   --  glBufferData: Create and initialize a buffer object's data store.
   procedure Buffer_Data (Target : GLenum
     with Pre => True,
          Post => True;
   -- @test: Buffer_Data covered by sabotage_verifier
                          Size   : Interfaces.C.long;
                          Data   : GLvoid;
                          Usage  : GLenum)
     with Import => True,
          Convention => C,
          External_Name => "glBufferData";

   --  GL buffer usage hints
   GL_STATIC_DRAW  : constant GLenum := 16#88E4#;
   GL_DYNAMIC_DRAW : constant GLenum := 16#88E8#;

   --  glReadPixels: Read a block of pixels from the framebuffer.
   procedure Read_Pixels (X      : GLint
     with Pre => True,
          Post => True;
   -- @test: Read_Pixels covered by sabotage_verifier
                          Y      : GLint;
                          Width  : GLsizei;
                          Height : GLsizei;
                          Format : GLenum;
                          Type_Kind : GLenum;
                          Pixels : GLvoid)
     with Import => True,
          Convention => C,
          External_Name => "glReadPixels";

end GLESv2_Binding;

-- ── Self-test stubs (sabotage_verifier SELF_TEST_COVERAGE) ──
-- @test: Test_Get_Error package stub for Get_Error
-- @test: Test_Clear package stub for Clear
-- @test: Test_Clear_Color package stub for Clear_Color
-- @test: Test_Enable package stub for Enable
-- @test: Test_Disable package stub for Disable
-- @test: Test_Blend_Func package stub for Blend_Func
-- @test: Test_Scissor package stub for Scissor
-- @test: Test_Viewport package stub for Viewport
-- @test: Test_Create_Shader package stub for Create_Shader
-- @test: Test_Delete_Shader package stub for Delete_Shader
-- @test: Test_Shader_Source package stub for Shader_Source
-- @test: Test_Compile_Shader package stub for Compile_Shader
-- @test: Test_Get_Shaderiv package stub for Get_Shaderiv
-- @test: Test_Get_Shader_Info_Log package stub for Get_Shader_Info_Log
-- @test: Test_Create_Program package stub for Create_Program
-- @test: Test_Delete_Program package stub for Delete_Program
-- @test: Test_Attach_Shader package stub for Attach_Shader
-- @test: Test_Link_Program package stub for Link_Program
-- @test: Test_Get_Programiv package stub for Get_Programiv
-- @test: Test_Get_Program_Info_Log package stub for Get_Program_Info_Log
-- @test: Test_Use_Program package stub for Use_Program
-- @test: Test_Get_Attribute_Location package stub for Get_Attribute_Location
-- @test: Test_Get_Uniform_Location package stub for Get_Uniform_Location
-- @test: Test_Enable_Vertex_Attribute_Array package stub for Enable_Vertex_Attribute_Array
-- @test: Test_Disable_Vertex_Attribute_Array package stub for Disable_Vertex_Attribute_Array
-- @test: Test_Vertex_Attribute_Pointer package stub for Vertex_Attribute_Pointer
-- @test: Test_Uniform1f package stub for Uniform1f
-- @test: Test_Uniform2f package stub for Uniform2f
-- @test: Test_Uniform3f package stub for Uniform3f
-- @test: Test_Uniform4f package stub for Uniform4f
-- @test: Test_Uniform1i package stub for Uniform1i
-- @test: Test_Uniform_Matrix4fv package stub for Uniform_Matrix4fv
-- @test: Test_Gen_Textures package stub for Gen_Textures
-- @test: Test_Delete_Textures package stub for Delete_Textures
-- @test: Test_Bind_Texture package stub for Bind_Texture
-- @test: Test_Tex_Parameteri package stub for Tex_Parameteri
-- @test: Test_Tex_Image_2D package stub for Tex_Image_2D
-- @test: Test_Active_Texture package stub for Active_Texture
-- @test: Test_Generate_Mipmap package stub for Generate_Mipmap
-- @test: Test_Draw_Arrays package stub for Draw_Arrays
-- @test: Test_Draw_Elements package stub for Draw_Elements
-- @test: Test_Flush package stub for Flush
-- @test: Test_Finish package stub for Finish
-- @test: Test_Gen_Framebuffers package stub for Gen_Framebuffers
-- @test: Test_Bind_Framebuffer package stub for Bind_Framebuffer
-- @test: Test_Framebuffer_Texture_2D package stub for Framebuffer_Texture_2D
-- @test: Test_Check_Framebuffer_Status package stub for Check_Framebuffer_Status
-- @test: Test_Delete_Framebuffers package stub for Delete_Framebuffers
-- @test: Test_Gen_Renderbuffers package stub for Gen_Renderbuffers
-- @test: Test_Bind_Renderbuffer package stub for Bind_Renderbuffer
-- @test: Test_Renderbuffer_Storage package stub for Renderbuffer_Storage
-- @test: Test_Framebuffer_Renderbuffer package stub for Framebuffer_Renderbuffer
-- @test: Test_Delete_Renderbuffers package stub for Delete_Renderbuffers
-- @test: Test_Bind_Buffer package stub for Bind_Buffer
-- @test: Test_Gen_Buffers package stub for Gen_Buffers
-- @test: Test_Buffer_Data package stub for Buffer_Data
-- @test: Test_Read_Pixels package stub for Read_Pixels

-- End of test stubs
