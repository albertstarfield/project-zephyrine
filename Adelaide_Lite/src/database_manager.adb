with Ada.Text_IO; use Ada.Text_IO;
with Ada_Sqlite3; use Ada_Sqlite3;
with Ada.Exceptions;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNATCOLL.JSON;
with Math_Utils;

package body Database_Manager is

   DB_File : constant String := "adelaide_memory.db";

   type DB_Access is access all Ada_Sqlite3.Database;
   Main_DB_Ptr : DB_Access := null;

   ----------------
   -- Initialize --
   ----------------
   procedure Initialize is
   begin
      if Main_DB_Ptr /= null then
         return;
      end if;

      Main_DB_Ptr := new Ada_Sqlite3.Database'(Open (DB_File));

      --  Memories table
      Execute (Main_DB_Ptr.all,
               "CREATE TABLE IF NOT EXISTS memories (" &
               "id INTEGER PRIMARY KEY AUTOINCREMENT," &
               "input TEXT," &
               "response TEXT," &
               "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)");

      --  Response Cache table (Semantic)
      Execute (Main_DB_Ptr.all,
               "CREATE TABLE IF NOT EXISTS response_cache (" &
               "id INTEGER PRIMARY KEY AUTOINCREMENT," &
               "prompt TEXT," &
               "embedding TEXT," &
               "response TEXT," &
               "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)");

      Put_Line ("[DB] Semantic Memory Core initialized.");
   exception
      when E : others =>
         Put_Line ("[DB] Critical Init Error: " &
           Ada.Exceptions.Exception_Message (E));
   end Initialize;

   ------------------
   -- Add_To_Cache --
   ------------------
   procedure Add_To_Cache
     (Prompt : String; Embedding : Math_Utils.Vector; Response : String)
   is
      use GNATCOLL.JSON;
      Vec_Obj : JSON_Array := Empty_Array;
   begin
      if Main_DB_Ptr = null then
         return;
      end if;

      for I in Embedding'Range loop
         Append (Vec_Obj, Create (Embedding (I)));
      end loop;

      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all,
            "INSERT INTO response_cache (prompt, embedding, response) " &
            "VALUES (?, ?, ?)");
      begin
         Bind_Text (Stmt, 1, Prompt);
         Bind_Text (Stmt, 2, Write (Create (Vec_Obj)));
         Bind_Text (Stmt, 3, Response);
         Step (Stmt);
      end;
   exception
      when others =>
         Put_Line ("[DB] Error updating response cache.");
   end Add_To_Cache;

   -------------------------
   -- Get_Cached_Response --
   -------------------------
   function Get_Cached_Response (Embedding : Math_Utils.Vector) return String is
      use GNATCOLL.JSON;
      Max_Sim  : Float := -2.0;
      Best_Res : Unbounded_String;
   begin
      if Main_DB_Ptr = null then
         return "";
      end if;

      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all, "SELECT embedding, response FROM response_cache");
      begin
         while Step (Stmt) = ROW loop
            declare
               Raw_Vec  : constant String := Column_Text (Stmt, 0);
               Raw_Resp : constant String := Column_Text (Stmt, 1);
               JSON_Vec : constant Read_Result := Read (Raw_Vec);
            begin
               if JSON_Vec.Success then
                  declare
                     Arr : constant JSON_Array := Get (JSON_Vec.Value);
                     Len : constant Natural := Length (Arr);
                     Entry_Vec : Math_Utils.Vector (1 .. Len);
                  begin
                     if Len = Embedding'Length then
                        for I in 1 .. Len loop
                           Entry_Vec (I) := Get (Get (Arr, I));
                        end loop;

                        declare
                           Sim : constant Float :=
                             Math_Utils.Cosine_Similarity
                               (Embedding, Entry_Vec);
                        begin
                           if Sim > Max_Sim then
                              Max_Sim := Sim;
                              if Sim >= 0.85 and then Sim < 0.98 then
                                 Best_Res := To_Unbounded_String (Raw_Resp);
                              end if;
                           end if;
                        end;
                     end if;
                  end;
               end if;
            end;
         end loop;
      end;

      return To_String (Best_Res);
   exception
      when others =>
         return "";
   end Get_Cached_Response;

   --------------
   -- Remember --
   --------------
   procedure Remember (User_Input : String; Assistant_Response : String) is
   begin
      if Main_DB_Ptr = null then
         return;
      end if;
      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all,
            "INSERT INTO memories (input, response) VALUES (?, ?)");
      begin
         Bind_Text (Stmt, 1, User_Input);
         Bind_Text (Stmt, 2, Assistant_Response);
         Step (Stmt);
      end;
   exception
      when others =>
         null;
   end Remember;

   ------------
   -- Recall --
   ------------
   function Recall (Query : String) return String is
      Result : Unbounded_String;
   begin
      if Main_DB_Ptr = null then
         return "";
      end if;
      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all,
            "SELECT response FROM memories WHERE input LIKE ? LIMIT 1");
      begin
         Bind_Text (Stmt, 1, "%" & Query & "%");
         if Step (Stmt) = ROW then
            Result := To_Unbounded_String (Column_Text (Stmt, 0));
         end if;
      end;
      return To_String (Result);
   exception
      when others =>
         return "";
   end Recall;

   -----------
   -- Close --
   -----------
   procedure Close is
   begin
      null;
   end Close;

end Database_Manager;
