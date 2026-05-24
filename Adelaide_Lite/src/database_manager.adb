with AnsiAda;
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
      if Lit_DB_Ptr = null then return; end if;

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
      if Lit_DB_Ptr = null then return; end if;

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
   -- Add_Graph_Relation --
   ------------------------
   procedure Add_Graph_Relation
     (Source   : String;
      Relation : String;
      Target   : String;
      Weight   : Float := 1.0)
   is
   begin
      if Lit_DB_Ptr = null then return; end if;
      declare
         Stmt : Statement := Prepare
           (Lit_DB_Ptr.all,
            "INSERT INTO knowledge_graph (source, relation, target, weight) " &
            "VALUES (?, ?, ?, ?)");
      begin
         Bind_Text (Stmt, 1, Source);
         Bind_Text (Stmt, 2, Relation);
         Bind_Text (Stmt, 3, Target);
         Bind_Double (Stmt, 4, Weight);
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
      if Main_DB_Ptr = null then return; end if;

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
      if Main_DB_Ptr = null then return ""; end if;

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
                              if Sim >= 0.85 and then Sim < 0.98 then
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
         if Best_Elapsed <= Float (2.0 * WCET) then
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
         else
            Execute (Main_DB_Ptr.all,
                    "UPDATE response_cache SET hit_count = 1, " &
                    "last_hit_time = CURRENT_TIMESTAMP WHERE id = " &
                    Best_Id'Img);
            return To_String (Best_Res);
         end if;
      end if;

      return "";
   exception
      when others => return "";
   end Get_Cached_Response;

   --------------
   -- Remember --
   --------------
   procedure Remember (Prompt : String; Response : String; Image_B64 : String := "") is
   begin
      if Main_DB_Ptr = null then return; end if;
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
      if Main_DB_Ptr = null then return ""; end if;
      declare
         Stmt : Statement := Prepare
           (Main_DB_Ptr.all,
            "SELECT id, response FROM memories WHERE input LIKE ? LIMIT 1");
      begin
         Bind_Text (Stmt, 1, "%" & Query & "%");
         if Step (Stmt) = ROW then
            Best_Id := Integer (Column_Int (Stmt, 0));
            Result := To_Unbounded_String (Column_Text (Stmt, 1));
         end if;
      end;
      
      if Best_Id /= -1 then
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
      if Main_DB_Ptr = null then return; end if;

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
      if Lit_DB_Ptr = null then return; end if;
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
      when others => if Is_Open (File) then Close (File); end if;
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
      if Lit_DB_Ptr = null then return; end if;

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

   procedure Close is
   begin
      null;
   end Close;

end Database_Manager;
