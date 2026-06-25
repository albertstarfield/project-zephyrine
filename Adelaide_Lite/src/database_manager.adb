pragma SPARK_Mode (Off);
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada_Sqlite3; use Ada_Sqlite3;
with Ada.Exceptions;
with Ada.Directories;
with GNATCOLL.JSON;
with Interfaces; use Interfaces;

package body Database_Manager is

   DB_File : constant String := "UI_Database/adelaide_memory.db";
   Lit_DB_File : constant String := "UI_Database/literatureRefIndex.db";

   type DB_Access is access all Ada_Sqlite3.Database;
   Main_DB_Ptr : DB_Access := null;
   Lit_DB_Ptr  : DB_Access := null;

   protected Init_Gate is
      procedure Do_Init;
   private
      Done : Boolean := False;
   end Init_Gate;

   protected body Init_Gate is
      procedure Do_Init is
      begin
         if Done then
            return;
         end if;

         if not Ada.Directories.Exists ("UI_Database") then
            Ada.Directories.Create_Directory ("UI_Database");
         end if;

         if Ada.Directories.Exists ("adelaide_memory.db") and then not Ada.Directories.Exists ("UI_Database/adelaide_memory.db") then
            Ada.Directories.Rename ("adelaide_memory.db", "UI_Database/adelaide_memory.db");
         end if;

         if Ada.Directories.Exists ("literatureRefIndex.db") and then not Ada.Directories.Exists ("UI_Database/literatureRefIndex.db") then
            Ada.Directories.Rename ("literatureRefIndex.db", "UI_Database/literatureRefIndex.db");
         end if;

         Main_DB_Ptr := new Ada_Sqlite3.Database'(Open (DB_File));

         --  Memories table
         Execute (Main_DB_Ptr.all,
                  "CREATE TABLE IF NOT EXISTS memories (" &
                  "id INTEGER PRIMARY KEY AUTOINCREMENT," &
                  "input TEXT," &
                  "response TEXT," &
                  "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP," &
                  "hit_count INTEGER DEFAULT 1," &
                  "last_hit_time DATETIME DEFAULT CURRENT_TIMESTAMP," &
                  "image_b64 TEXT)");

         --  System State table
         Execute (Main_DB_Ptr.all,
                  "CREATE TABLE IF NOT EXISTS system_state (" &
                  "key TEXT PRIMARY KEY," &
                  "value TEXT)");

         begin
            Execute (Main_DB_Ptr.all,
                    "ALTER TABLE memories ADD COLUMN hit_count " &
                    "INTEGER DEFAULT 1");
            Execute (Main_DB_Ptr.all,
                    "ALTER TABLE memories ADD COLUMN last_hit_time " &
                    "DATETIME DEFAULT CURRENT_TIMESTAMP");
            Execute (Main_DB_Ptr.all,
                    "ALTER TABLE memories ADD COLUMN image_b64 " &
                    "TEXT");
         exception
            when others => null; -- Columns already exist
         end;

         --  Response Cache table (Semantic)
         Execute (Main_DB_Ptr.all,
                  "CREATE TABLE IF NOT EXISTS response_cache (" &
                  "id INTEGER PRIMARY KEY AUTOINCREMENT," &
                  "prompt TEXT," &
                  "embedding TEXT," &
                  "response TEXT," &
                  "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP," &
                  "hit_count INTEGER DEFAULT 1," &
                  "last_hit_time DATETIME DEFAULT CURRENT_TIMESTAMP)");

         begin
            Execute (Main_DB_Ptr.all,
                    "ALTER TABLE response_cache ADD COLUMN hit_count " &
                    "INTEGER DEFAULT 1");
            Execute (Main_DB_Ptr.all,
                    "ALTER TABLE response_cache ADD COLUMN last_hit_time " &
                    "DATETIME DEFAULT CURRENT_TIMESTAMP");
          exception
             when others => null; -- Columns already exist
          end;

          --  LSH column for response_cache (QRNN speculation context)
          begin
             Execute (Main_DB_Ptr.all,
                     "ALTER TABLE response_cache ADD COLUMN lsh_hash " &
                     "INTEGER DEFAULT -1");
          exception
             when others => null; -- Column already exists
          end;

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

          --  LSH column for chunks (QRNN speculation context)
          begin
             Execute (Lit_DB_Ptr.all,
                     "ALTER TABLE chunks ADD COLUMN lsh_hash " &
                     "INTEGER DEFAULT -1");
          exception
             when others => null; -- Column already exists
          end;

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

         --  [VITAL-DO-NOT-REMOVE] Seed blacklist table.
         --  Persists blacklisted seeds across restarts.
         --  Stored in main DB (not .gitignored file).
         Execute (Main_DB_Ptr.all,
                  "CREATE TABLE IF NOT EXISTS seed_blacklist (" &
                  "seed INTEGER PRIMARY KEY," &
                  "reason TEXT DEFAULT 'think-only'," &
                  "blacklisted_at DATETIME DEFAULT CURRENT_TIMESTAMP)");

         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Imagined images table: stores images generated by ELP0 imagination tool.
         --  These are produced during Hybrid_Generate's cyclic background thoughts
         --  when the model calls [ACTION: imagine(prompt)].
         --  Stored with LSH hash for later retrieval by speculation context
         --  or VLM multimodal context injection.
         Execute (Main_DB_Ptr.all,
                  "CREATE TABLE IF NOT EXISTS imagined_images (" &
                  "id INTEGER PRIMARY KEY AUTOINCREMENT," &
                  "prompt TEXT," &
                  "image_b64 TEXT," &
                  "lsh_hash INTEGER DEFAULT -1," &
                  "width INTEGER DEFAULT 1024," &
                  "height INTEGER DEFAULT 1024," &
                  "seed INTEGER DEFAULT -1," &
                  "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)");

         begin
            Execute (Main_DB_Ptr.all,
                    "ALTER TABLE imagined_images ADD COLUMN width INTEGER DEFAULT 1024");
            Execute (Main_DB_Ptr.all,
                    "ALTER TABLE imagined_images ADD COLUMN height INTEGER DEFAULT 1024");
            Execute (Main_DB_Ptr.all,
                    "ALTER TABLE imagined_images ADD COLUMN seed INTEGER DEFAULT -1");
         exception
            when others => null; -- Columns already exist
         end;

         Done := True;
         Put_Line (AnsiAda.Foreground (AnsiAda.Magenta) & "[DB]" &
           AnsiAda.Reset & " Core initialized.");
      end Do_Init;
   end Init_Gate;

   ----------------
   -- Initialize --
   ----------------
   procedure Initialize is
   begin
      Init_Gate.Do_Init;
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Magenta) & "[DB]" &
           AnsiAda.Reset & " Critical Init Error: " &
           Ada.Exceptions.Exception_Message (E));
   end Initialize;

   ----------------------
   -- Set_System_State --
   ----------------------
   procedure Set_System_State (Key : String; Value : String) is
   begin
      if Main_DB_Ptr = null then
         return;
      end if;
      declare
         Stmt : Statement := Prepare (Main_DB_Ptr.all,
            "INSERT INTO system_state (key, value) VALUES (?, ?) " &
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value");
      begin
         Bind_Text (Stmt, 1, Key);
         Bind_Text (Stmt, 2, Value);
         Step (Stmt);
      end;
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Magenta) & "[DB]" &
           AnsiAda.Reset & " Set_System_State Error (" & Key & "): " &
           Ada.Exceptions.Exception_Message (E));
   end Set_System_State;

   ----------------------
   -- Get_System_State --
   ----------------------
   function Get_System_State (Key : String; Default : String := "") return String is
      Result : Unbounded_String := To_Unbounded_String (Default);
   begin
      if Main_DB_Ptr = null then
         return Default;
      end if;
      declare
         Stmt : Statement := Prepare (Main_DB_Ptr.all, "SELECT value FROM system_state WHERE key = ?");
      begin
         Bind_Text (Stmt, 1, Key);
         if Step (Stmt) = Row then
            Result := To_Unbounded_String (Column_Text (Stmt, 0));
         end if;
      end;
      return To_String (Result);
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Magenta) & "[DB]" &
           AnsiAda.Reset & " Get_System_State Error (" & Key & "): " &
           Ada.Exceptions.Exception_Message (E));
         return Default;
   end Get_System_State;

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
      when others => null;
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
                             Math_Utils.Cosine_Similarity (Embedding, Entry_Vec);
                        begin
                           if Sim >= 0.65 then
                              Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) &
                                                    "[Cache Match]" & AnsiAda.Reset &
                                                    " Score: " & Sim'Img & " | " &
                                                    "Source: " & Path_Str);
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
      when others => null;
   end Search_Literature;

   ------------------------
   -- Search_Interaction --
   ------------------------
   procedure Search_Interaction
     (Embedding : Math_Utils.Vector;
      Results   : out Chunk_Array;
      Count     : out Natural)
   is
      use GNATCOLL.JSON;
      Idx : Positive := Results'First;
   begin
      Count := 0;
      if Main_DB_Ptr = null then
         return;
      end if;

      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all, "SELECT prompt, response, embedding FROM response_cache");
      begin
         while Step (Stmt) = ROW and then Idx <= Results'Last loop
            declare
               Prompt_Str : constant String := Column_Text (Stmt, 0);
               Resp_Str   : constant String := Column_Text (Stmt, 1);
               Raw_Vec    : constant String := Column_Text (Stmt, 2);
               JSON_Vec   : constant Read_Result := Read (Raw_Vec);
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
                             Math_Utils.Cosine_Similarity (Embedding, Entry_Vec);
                        begin
                           if Sim >= 0.65 then
                              Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) &
                                                    "[Memory Match]" & AnsiAda.Reset &
                                                    " Score: " & Sim'Img);
                              Ada.Text_IO.Put_Line ("   -> User: " & Prompt_Str);
                              Ada.Text_IO.Put_Line ("   -> Adelaide: " & Resp_Str);
                              Results (Idx).File_Path :=
                                To_Unbounded_String ("Interaction");
                              Results (Idx).Content   :=
                                To_Unbounded_String ("User: " & Prompt_Str & ASCII.LF & "Adelaide: " & Resp_Str);
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
      when others => null;
   end Search_Interaction;

   ------------------------
   -- Add_Graph_Relation --
   ------------------------
   procedure Add_Graph_Relation
     (Source   : String;
      Relation : String;
      Target   : String;
      Weight   : Float := 1.0;
      Context  : String := "")
   is
   begin
      if Lit_DB_Ptr = null then
         return;
      end if;
      declare
         Stmt : Statement := Prepare
           (Lit_DB_Ptr.all,
            "INSERT INTO knowledge_graph (source, relation, target, weight, context) " &
            "VALUES (?, ?, ?, ?, ?)");
      begin
         Bind_Text (Stmt, 1, Source);
         Bind_Text (Stmt, 2, Relation);
         Bind_Text (Stmt, 3, Target);
         Bind_Double (Stmt, 4, Weight);
         Bind_Text (Stmt, 5, Context);
         Step (Stmt);
      end;
   exception
      when others => null;
   end Add_Graph_Relation;

   ------------------
   -- Add_To_Cache --
   ------------------
   procedure Add_To_Cache (Prompt : String;
                           Embedding : Math_Utils.Vector;
                           Response : String)
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
      when others => null;
   end Add_To_Cache;

   -------------------------
   -- Get_Cached_Response --
   -------------------------
   function Get_Cached_Response (Embedding : Math_Utils.Vector;
                                 WCET : Duration) return String
   is
      use GNATCOLL.JSON;
      Max_Sim : Float := -1.0;
      Best_Res : Unbounded_String;
      Best_Id : Integer := -1;
      Best_Hits : Integer := 0;
      Best_Elapsed : Float := 0.0;
   begin
      if Main_DB_Ptr = null then
         return "";
      end if;

      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all,
            "SELECT id, response, embedding, hit_count, " &
            "(strftime('%s','now') - strftime('%s', last_hit_time)) as elapsed " &
            "FROM response_cache");
      begin
         while Step (Stmt) = ROW loop
            declare
               Row_Id   : constant Integer := Column_Int (Stmt, 0);
               Raw_Resp : constant String := Column_Text (Stmt, 1);
               Raw_Vec  : constant String := Column_Text (Stmt, 2);
               Row_Hits : constant Integer := Column_Int (Stmt, 3);
               Elapsed  : constant Float := Column_Double (Stmt, 4);
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
                             Math_Utils.Cosine_Similarity (Embedding, Entry_Vec);
                        begin
                           if Sim > Max_Sim then
                              Max_Sim := Sim;
                              if Sim >= 0.90 then
                                 Best_Res := To_Unbounded_String (Raw_Resp);
                                 Best_Id := Row_Id;
                                 Best_Hits := Row_Hits;
                                 Best_Elapsed := Elapsed;
                              end if;
                           end if;
                        end;

                     end if;
                  end;
               end if;
            end;
         end loop;
      end;

      if Best_Id /= -1 then
         if Max_Sim >= 0.92 then
             Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) &
                                   "[Cache]" & AnsiAda.Reset &
                                   " Exact Match Hit ID" & Best_Id'Img);
         else
             Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) &
                                   "[Cache]" & AnsiAda.Reset &
                                   " Semantic Match (Sim: " & Max_Sim'Img & ") | ID" & Best_Id'Img);
         end if;

         --  Require at least 30s elapsed before serving from cache.
         --  Prevents stale/wrong responses from rapid re-requests.
         if Best_Elapsed < 30.0 then
            Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) &
                                  "[Cache]" & AnsiAda.Reset &
                                  " Too fresh (" & Best_Elapsed'Img &
                                  "s < 30s). Skipping cache.");
            return "";
         end if;

         --  Evict stale entries (hit 2+ times, served successfully)
         if Best_Hits >= 2 then
            Execute (Main_DB_Ptr.all,
                    "DELETE FROM response_cache WHERE id = " & Best_Id'Img);
            return "";
         else
            Execute (Main_DB_Ptr.all,
                    "UPDATE response_cache SET hit_count = hit_count + 1, " &
                    "last_hit_time = CURRENT_TIMESTAMP WHERE id = " &
                    Best_Id'Img);
            return To_String (Best_Res);
         end if;
      end if;

      return "";
   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[DB]" & AnsiAda.Reset &
                   " Cache Read Error: " & Ada.Exceptions.Exception_Message (E));
         return "";
   end Get_Cached_Response;

   --------------
   -- Remember --
   --------------
   procedure Remember (Prompt : String; Response : String; Image_B64 : String := "") is
   begin
      if Main_DB_Ptr = null then
         return;
      end if;
      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all,
            "INSERT INTO memories (input, response, image_b64) VALUES (?, ?, ?)");
      begin
         Bind_Text (Stmt, 1, Prompt);
         Bind_Text (Stmt, 2, Response);
         Bind_Text (Stmt, 3, Image_B64);
         Step (Stmt);
      end;
   exception
      when others => null;
   end Remember;

   ------------
   -- Recall --
   ------------
   function Recall (Query : String) return String is
      Result : Unbounded_String;
      Best_Id : Integer := -1;
   begin
      if Main_DB_Ptr = null then
         return "";
      end if;
      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all,
            "SELECT id, response FROM memories WHERE input LIKE ? LIMIT 1");
      begin
         Bind_Text (Stmt, 1, "%" & Query & "%");
         if Step (Stmt) = ROW then
            Best_Id := Column_Int (Stmt, 0);
            Result := To_Unbounded_String (Column_Text (Stmt, 1));
         end if;
      end;

      if Best_Id /= -1 then
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Green) &
                               "[Recall]" & AnsiAda.Reset &
                               " Memory hit: ID" & Best_Id'Img);
         Execute (Main_DB_Ptr.all,
                 "UPDATE memories SET hit_count = hit_count + 1, " &
                 "last_hit_time = CURRENT_TIMESTAMP WHERE id = " & Best_Id'Img);
      end if;
      return To_String (Result);
   exception
      when others => return "";
   end Recall;

   -------------------------
   -- Evict_Low_Salience --
   -------------------------
   procedure Evict_Low_Salience (Chunk_Size : Positive) is
      Alpha_Str : constant String := Alpha'Img;
   begin
      if Main_DB_Ptr = null then
         return;
      end if;

      Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[Salience]" &
                AnsiAda.Reset & " Evicting " & Chunk_Size'Img & " rows...");

      declare
         SQL : constant String :=
           "DELETE FROM response_cache WHERE id IN (" &
           "SELECT id FROM (" &
           "SELECT id, (hit_count / (1.0 + " & Alpha_Str &
           " * (strftime('%s','now') - strftime('%s', timestamp)))) as s " &
           "FROM response_cache ORDER BY s ASC LIMIT " & Chunk_Size'Img & "))";
      begin
         Execute (Main_DB_Ptr.all, SQL);
      end;

      declare
         SQL : constant String :=
           "DELETE FROM memories WHERE id IN (" &
           "SELECT id FROM (" &
           "SELECT id, (hit_count / (1.0 + " & Alpha_Str &
           " * (strftime('%s','now') - strftime('%s', timestamp)))) as s " &
           "FROM memories ORDER BY s ASC LIMIT " & Chunk_Size'Img & "))";
      begin
         Execute (Main_DB_Ptr.all, SQL);
      end;
   exception
      when others => null;
   end Evict_Low_Salience;

   ----------------
   -- Escape_XML --
   ----------------
   function Escape_XML (S : String) return String is
      Res : Unbounded_String;
   begin
      for I in S'Range loop
         case S (I) is
            when '<' => Append (Res, "&lt;");
            when '>' => Append (Res, "&gt;");
            when '&' => Append (Res, "&amp;");
            when '"' => Append (Res, "&quot;");
            when others => Append (Res, S (I));
         end case;
      end loop;
      return To_String (Res);
   end Escape_XML;

   --------------------
   -- Export_GraphML --
   --------------------
   procedure Export_GraphML (Filename : String) is
      File : File_Type;
   begin
      if Lit_DB_Ptr = null then
         return;
      end if;
      Create (File, Out_File, Filename);
      Put_Line (File, "<?xml version=""1.0"" encoding=""UTF-8""?>");
      Put_Line (File, "<graphml xmlns=""http://graphml.graphdrawing.org/xmlns"">");
      Put_Line (File, "  <key id=""d0"" for=""edge"" attr.name=""relation"" attr.type=""string""/>");
      Put_Line (File, "  <key id=""d1"" for=""edge"" attr.name=""weight"" attr.type=""double""/>");
      Put_Line (File, "  <graph id=""G"" edgedefault=""directed"">");

      declare
         Node_Stmt : Statement := Prepare
           (Lit_DB_Ptr.all,
            "SELECT DISTINCT node FROM (" &
            "SELECT source AS node FROM knowledge_graph " &
            "UNION " &
            "SELECT target AS node FROM knowledge_graph)");
      begin
         while Step (Node_Stmt) = ROW loop
            Put_Line (File, "    <node id=""" &
              Escape_XML (Column_Text (Node_Stmt, 0)) & """/>");
         end loop;
      end;

      declare
         Edge_Stmt : Statement := Prepare
           (Lit_DB_Ptr.all,
            "SELECT id, source, target, relation, weight FROM knowledge_graph");
      begin
         while Step (Edge_Stmt) = ROW loop
            declare
               Id_Val : constant String := Column_Text (Edge_Stmt, 0);
               Src    : constant String := Column_Text (Edge_Stmt, 1);
               Tgt    : constant String := Column_Text (Edge_Stmt, 2);
               Rel    : constant String := Column_Text (Edge_Stmt, 3);
               Wgt    : constant String := Column_Text (Edge_Stmt, 4);
            begin
               Put_Line (File, "    <edge id=""e" & Id_Val &
                         """ source=""" & Escape_XML (Src) &
                         """ target=""" & Escape_XML (Tgt) & """>");
               Put_Line (File, "      <data key=""d0"">" &
                         Escape_XML (Rel) & "</data>");
               Put_Line (File, "      <data key=""d1"">" & Wgt & "</data>");
               Put_Line (File, "    </edge>");
            end;
         end loop;
      end;

      Put_Line (File, "  </graph>");
      Put_Line (File, "</graphml>");
      Close (File);
   exception
      when others =>
         if Is_Open (File) then
            Close (File);
         end if;
   end Export_GraphML;

   ---------------------------------
   -- Get_Random_Literature_Chunk --
   ---------------------------------
   procedure Get_Random_Literature_Chunk
     (Content : out Unbounded_String;
      Success : out Boolean)
   is
   begin
      Success := False;
      Content := Null_Unbounded_String;
      if Lit_DB_Ptr = null then
         return;
      end if;

      declare
         Stmt : Statement := Prepare
           (Lit_DB_Ptr.all,
            "SELECT content FROM chunks ORDER BY RANDOM() LIMIT 1");
      begin
         if Step (Stmt) = ROW then
            Content := To_Unbounded_String (Column_Text (Stmt, 0));
            Success := True;
         end if;
      end;
   exception
      when others => null;
   end Get_Random_Literature_Chunk;

   -----------------------------
   -- Search_Interaction_By_LSH --
   -----------------------------
   procedure Search_Interaction_By_LSH
     (Hash      : Integer;
      Tolerance : Integer;
      Results   : out Chunk_Array;
      Count     : out Natural)
   is
      Idx : Positive := Results'First;
      --  Precompute all hash values within Hamming distance Tolerance
      --  For small Tolerance (0..3) this is fast; we generate candidates
      --  by flipping up to Tolerance bits.
      type Hash_Array is array (Positive range <>) of Integer;
      Max_Candidates : constant Positive := 1024; -- 2^10 = 1024 max, but we limit
      Candidates     : Hash_Array (1 .. 1024);
      NCand          : Natural := 0;
   begin
      Count := 0;
      if Main_DB_Ptr = null then
         return;
      end if;

      --  Generate all hashes within Hamming distance Tolerance.
      --  For Tolerance=2: 1 (exact) + 10 (1-bit) + 45 (2-bit) = 56 candidates.
      for Cand in 0 .. 1023 loop
         declare
            Dist : Natural := 0;
            V1   : Natural := Cand;
            V2   : Natural := Hash;
            Done : Boolean := False;
         begin
            for Bit in 0 .. 9 loop
               if (V1 mod 2) /= (V2 mod 2) then
                  Dist := Dist + 1;
                  if Dist > Tolerance then
                     Done := True;
                     exit;
                  end if;
               end if;
               V1 := V1 / 2;
               V2 := V2 / 2;
            end loop;
            if not Done then
               NCand := NCand + 1;
               Candidates (NCand) := Cand;
            end if;
         end;
      end loop;

      if NCand = 0 then
         return;
      end if;

      for C in 1 .. NCand loop
         if Idx > Results'Last then
            exit;
         end if;
         declare
            Inner_Stmt : Statement := Prepare
              (Main_DB_Ptr.all,
               "SELECT prompt, response FROM response_cache " &
               "WHERE lsh_hash = ? AND prompt IS NOT NULL " &
               "ORDER BY timestamp DESC LIMIT 1");
         begin
            Bind_Int (Inner_Stmt, 1, Candidates (C));
            if Step (Inner_Stmt) = ROW then
               declare
                  Prompt_Str : constant String := Column_Text (Inner_Stmt, 0);
                  Resp_Str   : constant String := Column_Text (Inner_Stmt, 1);
               begin
                  Results (Idx).File_Path :=
                    To_Unbounded_String ("Speculation:Interaction");
                  Results (Idx).Content   :=
                    To_Unbounded_String
                      ("User: " & Prompt_Str & ASCII.LF &
                       "Adelaide: " & Resp_Str);
                  Results (Idx).Score := 1.0;
                  Idx := Idx + 1;
                  Count := Count + 1;
               end;
            end if;
         end;
      end loop;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      if Count > 0 then
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Green) &
                   "[Memory][LSH]" & AnsiAda.Reset &
                   " Found " & Natural'Image (Count) &
                   " interaction(s) by LSH hash (Tolerance=" &
                   Integer'Image (Tolerance) & ").");
      end if;
   exception
      when others => null;
   end Search_Interaction_By_LSH;

   ----------------------------
   -- Search_Literature_By_LSH --
   ----------------------------
   procedure Search_Literature_By_LSH
     (Hash      : Integer;
      Tolerance : Integer;
      Results   : out Chunk_Array;
      Count     : out Natural)
   is
       Idx : Positive := Results'First;
       type Hash_Array is array (Positive range <>) of Integer;
       Max_Candidates : constant Positive := 1024;
       Candidates     : Hash_Array (1 .. 1024);
       NCand          : Natural := 0;
   begin
      Count := 0;
      if Lit_DB_Ptr = null then
         return;
      end if;

      --  Generate all hashes within Hamming distance Tolerance
      for Cand in 0 .. 1023 loop
         declare
            Dist : Natural := 0;
            V1   : Natural := Cand;
            V2   : Natural := Hash;
            Done : Boolean := False;
         begin
            for Bit in 0 .. 9 loop
               if (V1 mod 2) /= (V2 mod 2) then
                  Dist := Dist + 1;
                  if Dist > Tolerance then
                     Done := True;
                     exit;
                  end if;
               end if;
               V1 := V1 / 2;
               V2 := V2 / 2;
            end loop;
            if not Done then
               NCand := NCand + 1;
               Candidates (NCand) := Cand;
            end if;
         end;
      end loop;

      if NCand = 0 then
         return;
      end if;

      for C in 1 .. NCand loop
         if Idx > Results'Last then
            exit;
         end if;
         declare
            Inner_Stmt : Statement := Prepare
              (Lit_DB_Ptr.all,
               "SELECT file_path, content FROM chunks " &
               "WHERE lsh_hash = ? AND content IS NOT NULL " &
               "LIMIT 1");
         begin
            Bind_Int (Inner_Stmt, 1, Candidates (C));
            if Step (Inner_Stmt) = ROW then
               Results (Idx).File_Path :=
                 To_Unbounded_String (Column_Text (Inner_Stmt, 0));
               Results (Idx).Content   :=
                 To_Unbounded_String (Column_Text (Inner_Stmt, 1));
               Results (Idx).Score := 1.0;
               Idx := Idx + 1;
               Count := Count + 1;
            end if;
         end;
      end loop;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      if Count > 0 then
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Green) &
                   "[Memory][LSH]" & AnsiAda.Reset &
                   " Found " & Natural'Image (Count) &
                   " literature chunk(s) by LSH hash (Tolerance=" &
                   Integer'Image (Tolerance) & ").");
      end if;
   exception
      when others => null;
   end Search_Literature_By_LSH;

   --  =====================================================================
   --  SEED BLACKLIST (think-only prevention)
   --  =====================================================================
   --  When a seed produces only <think> with no visible content,
   --  it is blacklisted permanently. Generate skips blacklisted seeds.

   procedure Blacklist_Seed (Seed : Natural) is
   begin
      if Main_DB_Ptr = null then
         return;
      end if;
      declare
         Stmt : Statement := Prepare (Main_DB_Ptr.all,
                           "INSERT OR IGNORE INTO seed_blacklist (seed, reason) " &
                           "VALUES (?, 'think-only')");
      begin
         Bind_Text (Stmt, 1, Natural'Image (Seed));
         Step (Stmt);
         Put_Line
            (AnsiAda.Foreground (AnsiAda.Yellow)
             & "[Seed-BL]"
             & AnsiAda.Reset
             & " Blacklisted seed " & Natural'Image (Seed)
             & " (think-only response)");
      exception
         when others => null;  -- Non-fatal: blacklist is best-effort
      end;
   end Blacklist_Seed;

   function Is_Seed_Blacklisted (Seed : Natural) return Boolean is
      Result : Boolean := False;
   begin
      if Main_DB_Ptr = null then
         return False;
      end if;
      declare
         Stmt : Statement := Prepare (Main_DB_Ptr.all,
                           "SELECT COUNT(*) FROM seed_blacklist " &
                           "WHERE seed = ?");
      begin
         Bind_Text (Stmt, 1, Natural'Image (Seed));
         if Step (Stmt) = ROW then
            Result := Column_Int (Stmt, 0) > 0;
         end if;
      exception
         when others => Result := False;
      end;
      return Result;
   end Is_Seed_Blacklisted;

   function Get_Blacklist_Size return Natural is
      Count : Natural := 0;
   begin
      if Main_DB_Ptr = null then
         return 0;
      end if;
      declare
         Stmt : Statement := Prepare (Main_DB_Ptr.all,
                           "SELECT COUNT(*) FROM seed_blacklist");
      begin
         if Step (Stmt) = ROW then
            Count := Natural (Column_Int (Stmt, 0));
         end if;
      exception
         when others => Count := 0;
      end;
      return Count;
   end Get_Blacklist_Size;

   --  ============================================================================
   --  IMAGINED IMAGES: Store/retrieve images from ELP0 imagination
   --  ============================================================================
   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]

   procedure Store_Imagined_Image
     (Prompt    : String;
      Image_B64 : String;
      LSH_Hash  : Integer := -1)
   is
   begin
      if Main_DB_Ptr = null then
         return;
      end if;
      declare
         Stmt : Statement := Prepare (Main_DB_Ptr.all,
                        "INSERT INTO imagined_images (prompt, image_b64, lsh_hash) " &
                        "VALUES (?, ?, ?)");
      begin
         Bind_Text (Stmt, 1, Prompt);
         Bind_Text (Stmt, 2, Image_B64);
         Bind_Int (Stmt, 3, LSH_Hash);
         if Step (Stmt) /= DONE then
            Put_Line ("[DB] Store_Imagined_Image: INSERT failed");
         end if;
      exception
         when E : others =>
            Put_Line ("[DB] Store_Imagined_Image ERROR: " &
                      Ada.Exceptions.Exception_Message (E));
      end;
   end Store_Imagined_Image;

   procedure Search_Imagined_Images
     (Hash      : Integer;
      Tolerance : Integer;
      Results   : out Imagined_Image_Array;
      Count     : out Natural)
   is
      Max_Results : constant Positive := Results'Length;
   begin
      Count := 0;
      if Main_DB_Ptr = null then
         return;
      end if;

      --  Hamming distance within tolerance on 10-bit LSH hash
      --  We iterate all rows and check distance since SQLite has no bitwise XOR
      declare
         Stmt : Statement := Prepare (Main_DB_Ptr.all,
                        "SELECT prompt, image_b64, lsh_hash, created_at " &
                        "FROM imagined_images " &
                        "ORDER BY created_at DESC");
         Row_Count : Natural := 0;
         LSH_Dist  : Natural;
      begin
         while Step (Stmt) = ROW and then Row_Count < Max_Results loop
            declare
               Row_Hash : constant Natural := Natural (Column_Int (Stmt, 2));
               --  Hamming distance: count differing bits in 10-bit hash
               XOR_Val  : Natural;
               Dist     : Natural := 0;
               V        : Natural;
            begin
               XOR_Val := Natural (Unsigned_32 (Hash) xor Unsigned_32 (Row_Hash));
               V := XOR_Val;
               --  Brian Kernighan's bit counting
               while V > 0 loop
                  V := Natural (Unsigned_32 (V) and Unsigned_32 (V - 1));
                  Dist := Dist + 1;
               end loop;
               LSH_Dist := Dist;

               if LSH_Dist <= Tolerance then
                  Row_Count := Row_Count + 1;
                  Results (Row_Count) :=
                    (Image_B64  => To_Unbounded_String (Column_Text (Stmt, 1)),
                     Prompt     => To_Unbounded_String (Column_Text (Stmt, 0)),
                     LSH_Hash   => Integer (Column_Int (Stmt, 2)),
                     Created_At => To_Unbounded_String (Column_Text (Stmt, 3)));
               end if;
            end;
         end loop;
         Count := Row_Count;
      exception
         when E : others =>
            Put_Line ("[DB] Search_Imagined_Images ERROR: " &
                      Ada.Exceptions.Exception_Message (E));
            Count := 0;
      end;
   end Search_Imagined_Images;

   procedure Get_Recent_Imagined_Images
     (Max_Count : Positive;
      Results   : out Imagined_Image_Array;
      Count     : out Natural)
   is
      Max_Results : constant Positive := Integer'Min (Max_Count, Results'Length);
   begin
      Count := 0;
      if Main_DB_Ptr = null then
         return;
      end if;
      declare
         Stmt : Statement := Prepare (Main_DB_Ptr.all,
                        "SELECT prompt, image_b64, lsh_hash, created_at " &
                        "FROM imagined_images " &
                        "ORDER BY created_at DESC " &
                        "LIMIT " & Integer'Image (Max_Results));
         Row_Count : Natural := 0;
      begin
         while Step (Stmt) = ROW loop
            Row_Count := Row_Count + 1;
            Results (Row_Count) :=
              (Image_B64  => To_Unbounded_String (Column_Text (Stmt, 1)),
               Prompt     => To_Unbounded_String (Column_Text (Stmt, 0)),
               LSH_Hash   => Integer (Column_Int (Stmt, 2)),
               Created_At => To_Unbounded_String (Column_Text (Stmt, 3)));
         end loop;
         Count := Row_Count;
      exception
         when E : others =>
            Put_Line ("[DB] Get_Recent_Imagined_Images ERROR: " &
                      Ada.Exceptions.Exception_Message (E));
            Count := 0;
      end;
   end Get_Recent_Imagined_Images;

   procedure Close is
   begin
      null;
   end Close;

end Database_Manager;
