with open("src/kv_cache_manager.adb", "r") as f:
    content = f.read()

# Fix Has_Cache_Files (it doesn't have Session_ID)
content = content.replace("Cache_Dir(Session_ID)", "Cache_Dir")

# Add missing body for Cache_Exists
cache_exists_body = """
   function Cache_Exists (Model_ID : String; Session_ID : String := "") return Boolean is
   begin
      -- Just returning false for now since it's unused or not fully implemented before
      return False;
   end Cache_Exists;

   function Has_Cache_Files return Boolean is"""

content = content.replace("   function Has_Cache_Files return Boolean is", cache_exists_body)

with open("src/kv_cache_manager.adb", "w") as f:
    f.write(content)
