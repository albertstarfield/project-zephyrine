import re

with open("src/model_manager.ads", "r") as f:
    ads = f.read()

ads = re.sub(
    r'procedure Save_KV_Cache_To_SSD\n\s*\(Kind\s*:\s*Model_Type;\n\s*Tokens\s*:\s*System\.Address;\n\s*N_Tokens\s*:\s*Interfaces\.C\.size_t\);',
    '''procedure Save_KV_Cache_To_SSD
      (Kind     : Model_Type;
       Tokens   : System.Address;
       N_Tokens : Interfaces.C.size_t;
       Session_ID : String := "");''',
    ads
)

ads = re.sub(
    r'function Load_KV_Cache_From_SSD\n\s*\(Kind\s*:\s*Model_Type;\n\s*Tokens\s*:\s*out System\.Address;\n\s*N_Tokens\s*:\s*out Interfaces\.C\.size_t\) return Boolean;',
    '''function Load_KV_Cache_From_SSD
      (Kind     : Model_Type;
       Tokens   : out System.Address;
       N_Tokens : out Interfaces.C.size_t;
       Session_ID : String := "") return Boolean;''',
    ads
)

with open("src/model_manager.ads", "w") as f:
    f.write(ads)

with open("src/model_manager.adb", "r") as f:
    adb = f.read()

adb = re.sub(
    r'procedure Save_KV_Cache_To_SSD\n\s*\(Kind\s*:\s*Model_Type;\n\s*Tokens\s*:\s*System\.Address;\n\s*N_Tokens\s*:\s*Interfaces\.C\.size_t\) is',
    '''procedure Save_KV_Cache_To_SSD
       (Kind     : Model_Type;
        Tokens   : System.Address;
        N_Tokens : Interfaces.C.size_t;
        Session_ID : String := "") is''',
    adb
)

adb = re.sub(
    r'function Load_KV_Cache_From_SSD\n\s*\(Kind\s*:\s*Model_Type;\n\s*Tokens\s*:\s*out System\.Address;\n\s*N_Tokens\s*:\s*out Interfaces\.C\.size_t\) return Boolean is',
    '''function Load_KV_Cache_From_SSD
       (Kind     : Model_Type;
        Tokens   : out System.Address;
        N_Tokens : out Interfaces.C.size_t;
        Session_ID : String := "") return Boolean is''',
    adb
)

# In Save_KV_Cache_To_SSD
adb = re.sub(
    r'KV_Cache_Manager\.Save_To_SSD_Async\n\s*\(Context\s*=>\s*Models \(Kind\)\.Context,\n\s*Tokens\s*=>\s*Tokens,\n\s*N_Tokens\s*=>\s*N_Tokens,\n\s*Model_ID\s*=>\s*Kind\'Img\);',
    '''KV_Cache_Manager.Save_To_SSD_Async
               (Context  => Models (Kind).Context,
                Tokens   => Tokens,
                N_Tokens => N_Tokens,
                Model_ID => Kind'Img,
                Session_ID => Session_ID);''',
    adb
)

# In Load_KV_Cache_From_SSD
adb = re.sub(
    r'KV_Cache_Manager\.Load_From_SSD_Lazy\n\s*\(Context\s*=>\s*Models \(Kind\)\.Context,\n\s*Tokens\s*=>\s*Tokens,\n\s*N_Tokens\s*=>\s*N_Tokens,\n\s*Model_ID\s*=>\s*Kind\'Img\)',
    '''KV_Cache_Manager.Load_From_SSD_Lazy
               (Context  => Models (Kind).Context,
                Tokens   => Tokens,
                N_Tokens => N_Tokens,
                Model_ID => Kind'Img,
                Session_ID => Session_ID)''',
    adb
)

# Also fix the inline call inside Hybrid_Generate
# (KV_Cache_Manager.Save_To_SSD_Async around line 7036)
adb = re.sub(
    r'KV_Cache_Manager\.Save_To_SSD_Async\n\s*\(Context\s*=>\s*Models \(Kind\)\.Context,\n\s*Tokens\s*=>\s*Tokens\.all\'Address,\n\s*N_Tokens\s*=>\n\s*Interfaces\.C\.size_t \(N_Toks\),\n\s*Model_ID\s*=>\s*Kind\'Img\);',
    '''KV_Cache_Manager.Save_To_SSD_Async
                                       (Context  => Models (Kind).Context,
                                        Tokens   => Tokens.all'Address,
                                        N_Tokens =>
                                           Interfaces.C.size_t (N_Toks),
                                        Model_ID => Kind'Img,
                                        Session_ID => Session_ID);''',
    adb
)

# Check inline call to Load_From_SSD_Lazy inside Generate
adb = re.sub(
    r'KV_Cache_Manager\.Load_From_SSD_Lazy\n\s*\(Context\s*=>\s*Models \(Kind\)\.Context,\n\s*Tokens\s*=>\s*Tokens\.all\'Address,\n\s*N_Tokens\s*=>\n\s*Loaded_N_Tokens,\n\s*Model_ID\s*=>\s*Kind\'Img\)',
    '''KV_Cache_Manager.Load_From_SSD_Lazy
                                       (Context  => Models (Kind).Context,
                                        Tokens   => Tokens.all'Address,
                                        N_Tokens =>
                                           Loaded_N_Tokens,
                                        Model_ID => Kind'Img,
                                        Session_ID => Session_ID)''',
    adb
)

with open("src/model_manager.adb", "w") as f:
    f.write(adb)
