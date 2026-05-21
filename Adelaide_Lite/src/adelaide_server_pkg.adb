with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with AWS.Headers;
with AWS.Messages;
with AWS.Response.Set;
with AWS.Resources.Streams;
with GNATCOLL.JSON;
with Math_Utils;
with Model_Manager; use Model_Manager;
with Database_Manager;
with Ada.Calendar; use Ada.Calendar;
with Streaming_Queue;

package body Adelaide_Server_Pkg is

   subtype ID_Type is String (1 .. 64);
   type Entry_Rec is record
      ID  : ID_Type;
      Len : Natural;
      Q   : Streaming_Queue.Queue_Access;
   end record;
   type Map_Type is array (1 .. 100) of Entry_Rec;

   protected Stream_Registry is
      procedure Register (ID : String; Q : Streaming_Queue.Queue_Access);
      procedure Unregister (ID : String);
      procedure Push_Log (ID : String; Log : String);
   private
      Map   : Map_Type;
      Count : Natural := 0;
   end Stream_Registry;

   protected body Stream_Registry is
      procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) is
         S_ID : ID_Type := (others => ' ');
      begin
         if Count < 100 then
            Count := Count + 1;
            if ID'Length <= 64 then
               S_ID (1 .. ID'Length) := ID;
               Map (Count).Len := ID'Length;
            else
               S_ID := ID (ID'First .. ID'First + 63);
               Map (Count).Len := 64;
            end if;
            Map (Count).ID := S_ID;
            Map (Count).Q := Q;
         end if;
      end Register;

      procedure Unregister (ID : String) is
      begin
         for I in 1 .. Count loop
            if Map (I).ID (1 .. Map (I).Len) = ID then
               Map (I .. Count - 1) := Map (I + 1 .. Count);
               Count := Count - 1;
               return;
            end if;
         end loop;
      end Unregister;

      procedure Push_Log (ID : String; Log : String) is
      begin
         for I in 1 .. Count loop
            if Map (I).ID (1 .. Map (I).Len) = ID then
               Model_Manager.Push_Chunk (Map (I).Q, ID, Log);
               return;
            end if;
         end loop;
         Ada.Text_IO.Put_Line (Ada.Text_IO.Standard_Error, "[Log] " & ID & ": " & Log);
      end Push_Log;
   end Stream_Registry;

   procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) is
   begin
      Stream_Registry.Register (ID, Q);
   end Register;

   procedure Unregister (ID : String) is
   begin
      Stream_Registry.Unregister (ID);
   end Unregister;

   procedure Push_Log (ID : String; Log : String) is
   begin
      Stream_Registry.Push_Log (ID, Log);
   end Push_Log;

   task type Generator_Task is
      entry Start
        (Stream_Ptr     : Streaming_Queue.Queue_Access;
         Prompt_Val     : String;
         Images_Val     : GNATCOLL.JSON.JSON_Array;
         Session_ID_Val : String;
         URI_Str_Val    : String;
         Start_Time_Val : Ada.Calendar.Time;
         Level_Val      : Model_Manager.ELP_Level := Model_Manager.ELP1);
   end Generator_Task;

   type Generator_Task_Access is access Generator_Task;

   task body Generator_Task is
      Stream     : Streaming_Queue.Queue_Access;
      Prompt     : Unbounded_String;
      Images     : GNATCOLL.JSON.JSON_Array;
      Session_ID : Unbounded_String;
      Level      : Model_Manager.ELP_Level;
   begin
      accept Start
        (Stream_Ptr     : Streaming_Queue.Queue_Access;
         Prompt_Val     : String;
         Images_Val     : GNATCOLL.JSON.JSON_Array;
         Session_ID_Val : String;
         URI_Str_Val    : String;
         Start_Time_Val : Ada.Calendar.Time;
         Level_Val      : Model_Manager.ELP_Level := Model_Manager.ELP1) do
         Stream     := Stream_Ptr;
         Prompt     := To_Unbounded_String (Prompt_Val);
         Images     := Images_Val;
         Session_ID := To_Unbounded_String (Session_ID_Val);
         Level      := Level_Val;
      end Start;

      declare
         Res : Unbounded_String;
      begin
         Model_Manager.Hybrid_Generate
             (To_String (Prompt), Res, Images, To_String (Session_ID), Stream, Level);
         Stream_Registry.Unregister (To_String (Session_ID));
      exception
         when others =>
            Stream_Registry.Unregister (To_String (Session_ID));
      end;
   end Generator_Task;

   function Extract_Prompt (Body_Str : String) return String is
      use GNATCOLL.JSON;
      Res : constant Read_Result := Read (Body_Str);
   begin
      if not Res.Success or else Res.Value.Kind /= JSON_Object_Type then
         return "";
      end if;
      if Has_Field (Res.Value, "prompt") then
         return Get (Get (Res.Value, "prompt"));
      elsif Has_Field (Res.Value, "messages") then
         declare
            Arr : constant JSON_Array := Get (Get (Res.Value, "messages"));
         begin
            if Length (Arr) > 0 then
               declare
                  Last : constant JSON_Value := Get (Arr, Length (Arr));
               begin
                  if Has_Field (Last, "content") then
                     if Get (Last, "content").Kind = JSON_String_Type then
                        return Get (Get (Last, "content"));
                     elsif Get (Last, "content").Kind = JSON_Array_Type then
                        declare
                           C_Arr : constant JSON_Array := Get (Get (Last, "content"));
                           Acc : Unbounded_String;
                        begin
                           for I in 1 .. Length (C_Arr) loop
                              declare
                                 Item : constant JSON_Value := Get (C_Arr, I);
                              begin
                                 if Get (Item, "type") = Create ("text") then
                                    Append (Acc, String'(Get (Get (Item, "text"))));
                                 end if;
                              end;
                           end loop;
                           return To_String (Acc);
                        end;
                     end if;
                  end if;
               end;
            end if;
         end;
      end if;
      return "";
   end Extract_Prompt;

   function Extract_Images (Body_Str : String) return GNATCOLL.JSON.JSON_Array is
      use GNATCOLL.JSON;
      Res : constant Read_Result := Read (Body_Str);
   begin
      if not Res.Success then
         return Empty_Array;
      end if;
      if Has_Field (Res.Value, "images") then
         return Get (Get (Res.Value, "images"));
      elsif Has_Field (Res.Value, "messages") then
         declare
            Arr : constant JSON_Array := Get (Get (Res.Value, "messages"));
         begin
            if Length (Arr) > 0 then
               declare
                  Last : constant JSON_Value := Get (Arr, Length (Arr));
               begin
                  if Has_Field (Last, "content") and then 
                     Get (Last, "content").Kind = JSON_Array_Type 
                  then
                     declare
                        C_Arr : constant JSON_Array := Get (Get (Last, "content"));
                        R_Arr : JSON_Array := Empty_Array;
                     begin
                        for I in 1 .. Length (C_Arr) loop
                           declare
                              Item : constant JSON_Value := Get (C_Arr, I);
                           begin
                              if Get (Item, "type") = Create ("image_url") then
                                 Append (R_Arr, Get (Get (Item, "image_url"), "url"));
                              end if;
                           end;
                        end loop;
                        return R_Arr;
                     end;
                  end if;
               end;
            end if;
         end;
      end if;
      return Empty_Array;
   end Extract_Images;

   function Format_Universal_Response (URI : String; Text : String) return String is
      use GNATCOLL.JSON;
      Obj : constant JSON_Value := Create_Object;
   begin
      Set_Field (Obj, "model", String'("adelaide-hybrid"));
      Set_Field (Obj, "done", True);
      if URI = "/api/chat" or else URI = "/v1/chat/completions" then
         declare
            Msg : constant JSON_Value := Create_Object;
         begin
            Set_Field (Msg, "role", String'("assistant"));
            Set_Field (Msg, "content", Text);
            if URI = "/v1/chat/completions" then
               declare
                  C_Arr : JSON_Array := Empty_Array;
                  C_Obj : constant JSON_Value := Create_Object;
               begin
                  Set_Field (C_Obj, "message", Msg);
                  Append (C_Arr, C_Obj);
                  Set_Field (Obj, "choices", C_Arr);
               end;
            else
               Set_Field (Obj, "message", Msg);
            end if;
         end;
      else
         Set_Field (Obj, "response", Text);
      end if;
      return Write (Obj);
   end Format_Universal_Response;

   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data is
      use AWS.Status;
      URI_Str : constant String := URI (Request);
      Method_Val : constant Request_Method := Method (Request);
      Start_T : constant Time := Clock;
   begin
      begin
         if Method_Val = OPTIONS then
            declare
               R : AWS.Response.Data := AWS.Response.Acknowledge (AWS.Messages.S200);
            begin
               AWS.Response.Set.Add_Header (R, "Access-Control-Allow-Origin", "*");
               AWS.Response.Set.Add_Header (R, "Access-Control-Allow-Headers", 
                                            "Content-Type, Authorization, Session-ID");
               return R;
            end;
         end if;

         if Method_Val = GET then
            if URI_Str = "/v1/models" or else URI_Str = "/api/tags" then
               declare
                  use GNATCOLL.JSON;
                  Res : constant JSON_Value := Create_Object;
                  Arr : JSON_Array := Empty_Array;
                  M : constant JSON_Value := Create_Object;
                  D : constant JSON_Value := Create_Object;
               begin
                  Set_Field (M, "name", String'("adelaide-hybrid"));
                  Set_Field (M, "id", String'("adelaide-hybrid"));
                  Set_Field (D, "format", String'("gguf"));
                  Set_Field (D, "context_length", Create (Long_Long_Integer'(9_223_372_036_854_775_807)));
                  Set_Field (D, "embedding_length", Create (Long_Long_Integer'(4_294_967_295)));
                  Set_Field (M, "details", D);
                  Append (Arr, M);
                  if URI_Str = "/v1/models" then
                     Set_Field (Res, "data", Arr);
                  else
                     Set_Field (Res, "models", Arr);
                  end if;
                  return AWS.Response.Build (Content_Type => "application/json", 
                                             Message_Body => Write (Res));
               end;
            elsif URI_Str = "/api/version" then
               return AWS.Response.Build (Content_Type => "application/json", 
                                          Message_Body => "{""version"":""0.1.48""}");
            end if;
         end if;

         if Method_Val = POST then
            declare
               use GNATCOLL.JSON;
               B_Str : constant String := To_String (Binary_Data (Request));
               P_Raw : constant String := Extract_Prompt (B_Str);
               Prompt : Unbounded_String := To_Unbounded_String (P_Raw);
               Images : constant JSON_Array := Extract_Images (B_Str);
               SID_H : constant String := AWS.Headers.Get (AWS.Status.Header (Request), "Session-ID");
               SID : constant String := (if SID_H /= "" then SID_H else Peername (Request));
            begin
               if URI_Str = "/api/chat" or else URI_Str = "/v1/chat/completions" or else URI_Str = "/api/generate" then
                  if Length (Prompt) > 0 then
                     declare
                        use Database_Manager;
                        V : Math_Utils.Vector (1 .. 16384);
                        VL : Natural;
                        Ref_Res : Chunk_Array (1 .. 3);
                        NR : Natural;
                        Ctx : Unbounded_String;
                     begin
                        Model_Manager.Get_Embedding (To_String (Prompt), V, VL);
                        if VL > 0 then
                           Search_Literature (V (1 .. VL), Ref_Res, NR);
                           if NR > 0 then
                              Append (Ctx, "[CONTEXT]" & ASCII.LF);
                              for J in 1 .. NR loop
                                 Append (Ctx, To_String (Ref_Res (J).Content) & ASCII.LF);
                              end loop;
                              Prompt := Ctx & Prompt;
                           end if;
                        end if;
                     end;
                     
                     declare
                        Res_R : constant Read_Result := Read (B_Str);
                        Is_Str : Boolean := False;
                     begin
                        if Res_R.Success and then Has_Field (Res_R.Value, "stream") then
                           Is_Str := Get (Get (Res_R.Value, "stream"));
                        end if;
                        if Is_Str then
                           declare
                              Q : constant Streaming_Queue.Queue_Access := new Streaming_Queue.Queue;
                              T : constant Generator_Task_Access := new Generator_Task;
                              RS : constant Streaming_Queue.Response_Stream_Access := 
                                new Streaming_Queue.Response_Stream'(AWS.Resources.Streams.Stream_Type with Q => Q);
                           begin
                              Stream_Registry.Register (SID, Q);
                              T.Start (Q, To_String (Prompt), Images, SID, URI_Str, Start_T, ELP1);
                              return AWS.Response.Stream (Content_Type => "text/event-stream", Handle => RS);
                           end;
                        else
                           declare
                              G_Res : Unbounded_String;
                           begin
                              Hybrid_Generate (To_String (Prompt), G_Res, Images, SID, null, ELP1);
                              return AWS.Response.Build (Content_Type => "application/json", 
                                                         Message_Body => Format_Universal_Response (URI_Str, To_String (G_Res)));
                           end;
                        end if;
                     end;
                  end if;
               elsif URI_Str = "/api/embeddings" or else URI_Str = "/v1/embeddings" then
                  declare
                     V : Math_Utils.Vector (1 .. 16384);
                     VL : Natural;
                     Obj : constant JSON_Value := Create_Object;
                     A : JSON_Array := Empty_Array;
                  begin
                     Get_Embedding (To_String (Prompt), V, VL);
                     for I in 1 .. VL loop Append (A, Create (V (I))); end loop;
                     Set_Field (Obj, "embedding", A);
                     return AWS.Response.Build (Content_Type => "application/json", 
                                                Message_Body => Write (Obj));
                  end;
               end if;
            end;
         end if;
         return AWS.Response.Acknowledge (AWS.Messages.S404);
      exception
         when E : others =>
            return AWS.Response.Build (Content_Type => "application/json", 
                                       Message_Body => "{""error"":""Internal""}", 
                                       Status_Code => AWS.Messages.S500);
      end;
   end Dispatch;

end Adelaide_Server_Pkg;
