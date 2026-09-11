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
   procedure Clear_Color (Red   : GLfloat;
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
          External_Name => "glEnable";

   procedure Disable (Cap : GLenum)
     with Import => True,
          Convention => C,
          External_Name => "glDisable";

   --  glBlendFunc: Specify pixel arithmetic for blending.
   --  Citation: OpenGL ES 2.0 §4.1.7 "glBlendFunc defines the source and
   --  destination blending factors."
   procedure Blend_Func (Sfactor : GLenum;
                         Dfactor : GLenum)
     with Import => True,
          Convention => C,
          External_Name => "glBlendFunc";

   --  glScissor: Define the scissor rectangle.
   --  Citation: OpenGL ES 2.0 §4.2.5
   procedure Scissor (X      : GLint;
                      Y      : GLint;
                      Width  : GLsizei;
                      Height : GLsizei)
     with Import => True,
          Convention => C,
          External_Name => "glScissor";

   --  glViewport: Set the viewport transformation parameters.
   --  Citation: OpenGL ES 2.0 §2.12.1 "glViewport sets the viewport...
   --  x, y specify the lower-left corner; width, height specify size."
   procedure Viewport (X      : GLint;
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
   procedure Shader_Source (Shader     : GLuint;
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
   procedure Get_Shaderiv (Shader : GLuint;
                           Pname  : GLenum;
                           Params : access GLint)
     with Import => True,
          Convention => C,
          External_Name => "glGetShaderiv";

   --  glGetShaderInfoLog: Return the information log for a shader.
   procedure Get_Shader_Info_Log (Shader      : GLuint;
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
   procedure Attach_Shader (Program : GLuint;
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
   procedure Get_Programiv (Program : GLuint;
                            Pname   : GLenum;
                            Params  : access GLint)
     with Import => True,
          Convention => C,
          External_Name => "glGetProgramiv";

   --  glGetProgramInfoLog: Return the information log for a program.
   procedure Get_Program_Info_Log (Program   : GLuint;
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
   function Get_Attribute_Location (Program : GLuint;
                                    Name    : Interfaces.C.Strings.chars_ptr)
      return GLint
     with Import => True,
          Convention => C,
          External_Name => "glGetAttribLocation";

   --  glGetUniformLocation: Return the location of a uniform variable.
   function Get_Uniform_Location (Program : GLuint;
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

   procedure Disable_Vertex_Attribute_Array (Index : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDisableVertexAttribArray";

   --  glVertexAttribPointer: Define an array of vertex attribute data.
   --  Citation: OpenGL ES 2.0 §2.8 "glVertexAttribPointer specifies the
   --  location and data format of an array of generic attributes."
   procedure Vertex_Attribute_Pointer (Index      : GLuint;
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
   procedure Uniform1f (Location : GLint;
                        V0       : GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glUniform1f";

   --  glUniform2f: Set a vec2 uniform.
   procedure Uniform2f (Location : GLint;
                        V0, V1   : GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glUniform2f";

   --  glUniform3f: Set a vec3 uniform.
   procedure Uniform3f (Location : GLint;
                        V0, V1, V2 : GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glUniform3f";

   --  glUniform4f: Set a vec4 uniform (used for RGBA colors).
   procedure Uniform4f (Location    : GLint;
                        V0, V1, V2, V3 : GLfloat)
     with Import => True,
          Convention => C,
          External_Name => "glUniform4f";

   --  glUniform1i: Set a single int uniform (used for texture units).
   procedure Uniform1i (Location : GLint;
                        V0       : GLint)
     with Import => True,
          Convention => C,
          External_Name => "glUniform1i";

   --  glUniformMatrix4fv: Set a mat4 uniform (used for transforms).
   procedure Uniform_Matrix4fv (Location : GLint;
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
   procedure Gen_Textures (N      : GLsizei;
                           Textures : access GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glGenTextures";

   --  glDeleteTextures: Delete texture objects.
   procedure Delete_Textures (N        : GLsizei;
                              Textures : access constant GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDeleteTextures";

   --  glBindTexture: Bind a named texture to a target.
   procedure Bind_Texture (Target  : GLenum;
                           Texture : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glBindTexture";

   --  glTexParameteri: Set texture parameter (filtering, wrapping).
   procedure Tex_Parameteri (Target : GLenum;
                             Pname  : GLenum;
                             Param  : GLint)
     with Import => True,
          Convention => C,
          External_Name => "glTexParameteri";

   --  glTexImage2D: Specify a 2D texture image.
   --  Citation: OpenGL ES 2.0 §4.3.2 "glTexImage2D defines a 2D texture
   --  image... pixels is the image data in client memory."
   procedure Tex_Image_2D (Target     : GLenum;
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
   procedure Draw_Arrays (Mode  : GLenum;
                          First : GLint;
                          Count : GLsizei)
     with Import => True,
          Convention => C,
          External_Name => "glDrawArrays";

   --  glDrawElements: Render primitives from indexed array data.
   --  Citation: OpenGL ES 2.0 §3.8.2 "glDrawElements renders a sequence
   --  of geometric primitives using vertex indices."
   procedure Draw_Elements (Mode   : GLenum;
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
   procedure Gen_Framebuffers (N           : GLsizei;
                               Framebuffers : access GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glGenFramebuffers";

   --  glBindFramebuffer: Bind a framebuffer object.
   procedure Bind_Framebuffer (Target     : GLenum;
                               Framebuffer : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glBindFramebuffer";

   --  glFramebufferTexture2D: Attach a 2D texture to a framebuffer.
   procedure Framebuffer_Texture_2D (Target      : GLenum;
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
   procedure Delete_Framebuffers (N             : GLsizei;
                                  Framebuffers : access constant GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDeleteFramebuffers";

   --  glGenRenderbuffers: Generate renderbuffer object names.
   procedure Gen_Renderbuffers (N           : GLsizei;
                                Renderbuffers : access GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glGenRenderbuffers";

   --  glBindRenderbuffer: Bind a renderbuffer object.
   procedure Bind_Renderbuffer (Target      : GLenum;
                                Renderbuffer : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glBindRenderbuffer";

   --  glRenderbufferStorage: Allocate storage for a renderbuffer.
   procedure Renderbuffer_Storage (Target          : GLenum;
                                   Internal_Format : GLenum;
                                   Width           : GLsizei;
                                   Height          : GLsizei)
     with Import => True,
          Convention => C,
          External_Name => "glRenderbufferStorage";

   --  glFramebufferRenderbuffer: Attach a renderbuffer to a framebuffer.
   procedure Framebuffer_Renderbuffer (Target     : GLenum;
                                       Attachment : GLenum;
                                       Renderbuf_Target : GLenum;
                                       Renderbuffer : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glFramebufferRenderbuffer";

   --  glDeleteRenderbuffers: Delete renderbuffer objects.
   procedure Delete_Renderbuffers (N             : GLsizei;
                                   Renderbuffers : access constant GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glDeleteRenderbuffers";

   --  glBindBuffer: Bind a buffer object (VBO support).
   procedure Bind_Buffer (Target : GLenum;
                          Buffer : GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glBindBuffer";

   --  glGenBuffers: Generate buffer object names.
   procedure Gen_Buffers (N       : GLsizei;
                          Buffers : access GLuint)
     with Import => True,
          Convention => C,
          External_Name => "glGenBuffers";

   --  glBufferData: Create and initialize a buffer object's data store.
   procedure Buffer_Data (Target : GLenum;
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
   procedure Read_Pixels (X      : GLint;
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
