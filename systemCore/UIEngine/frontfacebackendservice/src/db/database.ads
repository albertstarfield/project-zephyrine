with Models;

package Database is

   procedure Initialize (DB_Path : String := "frontface.db");

   -- Chat Operations
   function Create_Chat (Title : String) return String;
   procedure Save_Message (Chat_ID : String; Role : String; Content : String);
   function Get_Chat_History (Chat_ID : String) return Models.Message_Vectors.Vector;
   function Get_All_Chats return Models.Chat_Vectors.Vector;

   -- NEW: File Operations
   procedure Store_File_Record 
     (Filename, Filetype, User_ID, LLM_ID, Status : String);
     
   function Get_Files (User_ID : String) return Models.File_Vectors.Vector;

end Database;