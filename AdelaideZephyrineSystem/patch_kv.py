import re

with open("src/kv_cache_manager.adb", "r") as f:
    content = f.read()

# 1. Update Cache_Dir definition
content = re.sub(
    r'function Cache_Dir return String is\n\s*begin\n\s*return "cache/kv/" & Get_User & "/";\n\s*end Cache_Dir;',
    '''function Cache_Dir (Session_ID : String := "") return String is
   begin
      if Session_ID /= "" then
         return "cache/kv/" & Session_ID & "/";
      end if;
      return "cache/kv/" & Get_User & "/";
   end Cache_Dir;''',
    content
)

# 2. Update Save_To_SSD_Async signature
content = re.sub(
    r'procedure Save_To_SSD_Async\n\s*\(Context\s*:\s*Llama_Interface\.Llama_Context;\n\s*Tokens\s*:\s*System\.Address;\n\s*N_Tokens\s*:\s*Interfaces\.C\.size_t;\n\s*Model_ID\s*:\s*String\)',
    '''procedure Save_To_SSD_Async
      (Context    : Llama_Interface.Llama_Context;
       Tokens     : System.Address;
       N_Tokens   : Interfaces.C.size_t;
       Model_ID   : String;
       Session_ID : String := "")''',
    content
)

# 3. Update Load_From_SSD_Lazy signature
content = re.sub(
    r'function Load_From_SSD_Lazy\n\s*\(Context\s*:\s*Llama_Interface\.Llama_Context;\n\s*Tokens\s*:\s*out System\.Address;\n\s*N_Tokens\s*:\s*out Interfaces\.C\.size_t;\n\s*Model_ID\s*:\s*String\) return Boolean',
    '''function Load_From_SSD_Lazy
      (Context    : Llama_Interface.Llama_Context;
       Tokens     : out System.Address;
       N_Tokens   : out Interfaces.C.size_t;
       Model_ID   : String;
       Session_ID : String := "") return Boolean''',
    content
)

# 4. Update Cache_Exists signature
content = re.sub(
    r'function Cache_Exists \(Model_ID : String\) return Boolean',
    '''function Cache_Exists (Model_ID : String; Session_ID : String := "") return Boolean''',
    content
)

# 5. Fix internal calls to Cache_Dir inside these modified functions to pass Session_ID
# In Save_To_SSD_Async (approx line 778)
content = re.sub(
    r'File_Path\s*:\s*constant String := Cache_Dir & Model_ID',
    r'File_Path   : constant String := Cache_Dir(Session_ID) & Model_ID',
    content
)

# In Cache_Exists (approx line 1041)
content = re.sub(
    r'if not Exists \(Cache_Dir\) then',
    r'if not Exists (Cache_Dir(Session_ID)) then',
    content
)
content = re.sub(
    r'Search \(Cache_Dir, "\*\.bin"',
    r'Search (Cache_Dir(Session_ID), "*.bin"',
    content
)

with open("src/kv_cache_manager.adb", "w") as f:
    f.write(content)
