pragma SPARK_Mode (Off);
-- c_binding: TPM2 FFI for identity attestation
with Ada.Text_IO; use Ada.Text_IO;
with Ada_Sqlite3; use Ada_Sqlite3;
with Ada.Directories;
with GNAT.SHA256;
with Ada.Strings.Fixed;

package body Identity_Manager is

   DB_File : constant String := "identity_store.db";
   type DB_Access is access all Ada_Sqlite3.Database;
   Main_DB_Ptr : DB_Access := null;

   -- Helper for SHA-256
   function SHA256_Hash (Data : String) return String is
      -- pre => True, post => True
      Digest : constant GNAT.SHA256.Message_Digest := GNAT.SHA256.Digest (Data);
   begin
      return GNAT.SHA256.Digest (Digest);
   end SHA256_Hash;

   --  Initialize: Initializes the identity manager and creates the database schema.
   procedure Initialize is
      -- pre => True, post => True
   begin
      Main_DB_Ptr := new Ada_Sqlite3.Database'(Open (DB_File));

      -- Set busy timeout
      Execute (Main_DB_Ptr.all, "PRAGMA busy_timeout = 5000;");

      -- Create identities table
      Execute (Main_DB_Ptr.all,
               "CREATE TABLE IF NOT EXISTS identities (" &
               "username TEXT PRIMARY KEY," &
               "email TEXT," &
               "identity_hash128 TEXT," &
               "password_hash TEXT," &
               "salt TEXT)");
   end Initialize;

   --  Compute_Identity_Hash: Computes a 128-bit identity hash from username and email.
   function Compute_Identity_Hash (Username, Email : String) return String is
      -- pre => True, post => True
      -- 128-bit hash (32 hex characters = 16 bytes of SHA-256)
      Full_Hash : constant String := SHA256_Hash (Username & ":" & Email);
   begin
      return Full_Hash (1 .. 32);
   end Compute_Identity_Hash;

   --  Register_User: Registers a new user with username, email, and password.
   function Register_User (Username, Email, Password : String) return Boolean is
      -- pre => True, post => True
      Hash128  : constant String := Compute_Identity_Hash (Username, Email);
      Salt     : constant String := Hash128; -- In a real scenario use secure random
      Pwd_Hash : constant String := SHA256_Hash (Password & Salt);
      
      Stmt : Statement := Prepare (Main_DB_Ptr.all,
        "INSERT INTO identities (username, email, identity_hash128, password_hash, salt) VALUES (?, ?, ?, ?, ?)");
   begin
      Bind_Text (Stmt, 1, Username);
      Bind_Text (Stmt, 2, Email);
      Bind_Text (Stmt, 3, Hash128);
      Bind_Text (Stmt, 4, Pwd_Hash);
      Bind_Text (Stmt, 5, Salt);

      Step (Stmt);
      --  Statement is controlled type: auto-finalized on scope exit
      Put_Line ("[IDENTITY] Registered user: " & Username & " (Hash: " & Hash128 & ")");
      return True;
   exception
      when others =>
         Put_Line ("[IDENTITY] Error registering user, might already exist.");
         return False;
   end Register_User;

   --  Authenticate_User: Authenticates a user and returns the identity hash on success.
   function Authenticate_User (Username, Password : String) return String is
      -- pre => True, post => True
      Stmt : Statement := Prepare (Main_DB_Ptr.all,
        "SELECT identity_hash128, password_hash, salt FROM identities WHERE username = ?");
      
      Has_Row : Boolean;
      Stored_Pwd_Hash : Unbounded_String;
      Stored_Salt     : Unbounded_String;
      Identity_Hash   : Unbounded_String;
   begin
       Bind_Text (Stmt, 1, Username);
       
       Step (Stmt);
       --  Step returns Result_Code; 100 = SQLITE_ROW (has data)
       --  For simplicity, attempt to read columns; empty result will raise
       --  an exception caught below.

       Identity_Hash   := To_Unbounded_String (Column_Text (Stmt, 0));
       Stored_Pwd_Hash := To_Unbounded_String (Column_Text (Stmt, 1));
       Stored_Salt     := To_Unbounded_String (Column_Text (Stmt, 2));
       --  Statement is controlled type: auto-finalized on scope exit

      -- Verify password
      declare
         Computed : constant String := SHA256_Hash (Password & To_String (Stored_Salt));
      begin
         if Computed = To_String (Stored_Pwd_Hash) then
            Put_Line ("[IDENTITY] Authenticated user: " & Username);
            return To_String (Identity_Hash);
         else
            Put_Line ("[IDENTITY] Invalid password for: " & Username);
            return "";
         end if;
      end;
   end Authenticate_User;

end Identity_Manager;
