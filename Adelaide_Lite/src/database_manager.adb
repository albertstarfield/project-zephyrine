with Ada.Text_IO; use Ada.Text_IO;
with Ada_Sqlite3; use Ada_Sqlite3;
with Ada.Exceptions;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNATCOLL.JSON;
with Math_Utils;

package body Database_Manager is

   DB_File : constant String := "adelaide_memory.db";
   Lit_DB_File : constant String := "literatureRefIndex.db";

   type DB_Access is access all Ada_Sqlite3.Database;
   Main_DB_Ptr : DB_Access := null;
   Lit_DB_Ptr  : DB_Access := null;

   ----------------
   -- Initialize --
   ----------------
   procedure Initialize is
   begin
      if Main_DB_Ptr = null then
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
      end if;

      if Lit_DB_Ptr = null then
         Lit_DB_Ptr := new Ada_Sqlite3.Database'(Open (Lit_DB_File));

         --  Chunks table for literature
         Execute (Lit_DB_Ptr.all,
                  "CREATE TABLE IF NOT EXISTS chunks (" &
                  "id INTEGER PRIMARY KEY AUTOINCREMENT," &
                  "file_path TEXT," &
                  "content TEXT," &
                  "embedding TEXT," &
                  "hash TEXT," &
                  "indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP)");

         --  Graph table for relationships
         Execute (Lit_DB_Ptr.all,
                  "CREATE TABLE IF NOT EXISTS knowledge_graph (" &
                  "id INTEGER PRIMARY KEY AUTOINCREMENT," &
                  "source TEXT," &
                  "relation TEXT," &
                  "target TEXT," &
                  "weight REAL," &
                  "context TEXT," &
                  "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");
      end if;

      Put_Line ("[DB] Semantic Memory and Literature Core initialized.");
   exception
      when E : others =>
         Put_Line ("[DB] Critical Init Error: " &
           Ada.Exceptions.Exception_Message (E));
   end Initialize;

   --------------------------
   -- Add_Literature_Chunk --
   --------------------------
   procedure Add_Literature_Chunk
     (File_Path : String; 
      Content   : String; 
      Embedding : Math_Utils.Vector;
      Doc_Hash  : String)
   is
      use GNATCOLL.JSON;
      Vec_Obj : JSON_Array := Empty_Array;
   begin
      if Lit_DB_Ptr = null then
         return;
      end if;

      for I in Embedding'Range loop
         Append (Vec_Obj, Create (Embedding (I)));
      end loop;

      declare
         Stmt : Statement := Prepare
           (Lit_DB_Ptr.all,
            "INSERT INTO chunks (file_path, content, embedding, hash) " &
            "VALUES (?, ?, ?, ?)");
      begin
         Bind_Text (Stmt, 1, File_Path);
         Bind_Text (Stmt, 2, Content);
         Bind_Text (Stmt, 3, Write (Create (Vec_Obj)));
         Bind_Text (Stmt, 4, Doc_Hash);
         Step (Stmt);
      end;
   exception
      when others =>
         Put_Line ("[DB] Error adding literature chunk.");
   end Add_Literature_Chunk;

   -----------------------
   -- Search_Literature --
   -----------------------
   procedure Search_Literature
     (Embedding : Math_Utils.Vector;
      Results   : out Chunk_Array;
      Count     : out Natural)
   is
      use GNATCOLL.JSON;
      Idx : Positive := Results'First;
   begin
      Count := 0;
      if Lit_DB_Ptr = null then
         return;
      end if;

      declare
         Stmt : Statement := Prepare
           (Lit_DB_Ptr.all, "SELECT file_path, content, embedding FROM chunks");
      begin
         while Step (Stmt) = ROW and then Idx <= Results'Last loop
            declare
               Path_Str : constant String := Column_Text (Stmt, 0);
               Text_Str : constant String := Column_Text (Stmt, 1);
               Raw_Vec  : constant String := Column_Text (Stmt, 2);
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
                           if Sim >= 0.65 then
                              Results (Idx).File_Path := 
                                To_Unbounded_String (Path_Str);
                              Results (Idx).Content   := 
                                To_Unbounded_String (Text_Str);
                              Results (Idx).Score     := Sim;
                              Idx := Idx + 1;
                              Count := Count + 1;
                           end if;
                        end;
                     end if;
                  end;
               end if;
            end;
         end loop;
      end;
   exception
      when others =>
         null;
   end Search_Literature;

   ------------------------
   -- Add_Graph_Relation --
   ------------------------
   procedure Add_Graph_Relation
     (Source   : String;
      Relation : String;
      Target   : String;
      Weight   : Float := 1.0)
   is
   begin
      if Lit_DB_Ptr = null then
         return;
      end if;
      declare
         Stmt : Statement := Prepare
           (Lit_DB_Ptr.all,
            "INSERT INTO knowledge_graph (source, relation, target, weight) " &
            "VALUES (?, ?, ?, ?)");
      begin
         Bind_Text (Stmt, 1, Source);
         Bind_Text (Stmt, 2, Relation);
         Bind_Text (Stmt, 3, Target);
         Bind_Text (Stmt, 4, Weight'Img);
         Step (Stmt);
      end;
   exception
      when others =>
         null;
   end Add_Graph_Relation;

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
