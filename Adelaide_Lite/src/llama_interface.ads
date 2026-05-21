with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

package Llama_Interface is
   pragma Spark_Mode (Off);

   type Llama_Model is new System.Address;
   type Llama_Context is new System.Address;
   type Llama_Vocab is new System.Address;
   type Llama_Sampler is new System.Address;

   type Llama_Token is new int;
   type Llama_Pos is new int;
   type Llama_Seq_Id is new int;

   Null_Model   : constant Llama_Model := Llama_Model (System.Null_Address);
   Null_Context : constant Llama_Context :=
     Llama_Context (System.Null_Address);

   type Llama_Token_Data is record
      Id    : Llama_Token;
      Logit : Float;
      P     : Float;
   end record;
   pragma Convention (C, Llama_Token_Data);

   type Llama_Token_Data_Access is access all Llama_Token_Data;

   type Llama_Token_Data_Array is record
      Data     : Llama_Token_Data_Access;
      Size     : size_t;
      Selected : Long_Long_Integer;
      Sorted   : Boolean;
   end record;
   pragma Convention (C, Llama_Token_Data_Array);

   --  Minimal set of params to get started
   type Llama_Model_Params is record
      Devices                    : System.Address;
      Tensor_Buft_Overrides      : System.Address;
      N_Gpu_Layers               : int;
      Split_Mode                 : int;
      Main_Gpu                   : int;
      Tensor_Split               : System.Address;
      Progress_Callback          : System.Address;
      Progress_Callback_User_Data : System.Address;
      Kv_Overrides               : System.Address;
      Vocab_Only                 : Boolean;
      Use_Mmap                   : Boolean;
      Use_Direct_Io              : Boolean;
      Use_Mlock                  : Boolean;
      Check_Tensors              : Boolean;
      Use_Extra_Bufts            : Boolean;
      No_Host                    : Boolean;
      No_Alloc                   : Boolean;
   end record;
   pragma Convention (C, Llama_Model_Params);

   type Llama_Context_Params is record
      N_Ctx           : unsigned;
      N_Batch         : unsigned;
      N_Ubatch        : unsigned;
      N_Seq_Max       : unsigned;
      N_Rs_Seq        : unsigned;
      N_Threads       : int;
      N_Threads_Batch : int;

      Ctx_Type           : int;
      Rope_Scaling_Type  : int;
      Pooling_Type       : int;
      Attention_Type     : int;
      Flash_Attn_Type    : int;

      Rope_Freq_Base   : Float;
      Rope_Freq_Scale  : Float;
      Yarn_Ext_Factor  : Float;
      Yarn_Attn_Factor : Float;
      Yarn_Beta_Fast   : Float;
      Yarn_Beta_Slow   : Float;
      Yarn_Orig_Ctx    : unsigned;
      Defrag_Thold     : Float;

      Cb_Eval           : System.Address;
      Cb_Eval_User_Data : System.Address;

      Type_K : int;
      Type_V : int;

      Abort_Callback      : System.Address;
      Abort_Callback_Data : System.Address;

      Embeddings  : Boolean;
      Offload_Kqv : Boolean;
      No_Perf     : Boolean;
      Op_Offload  : Boolean;
      Swa_Full    : Boolean;
      Kv_Unified  : Boolean;

      Samplers    : System.Address;
      N_Samplers  : size_t;
   end record;
   pragma Convention (C, Llama_Context_Params);

   type Llama_Batch is record
      N_Tokens : int;
      Token    : System.Address;
      Embd     : System.Address;
      Pos      : System.Address;
      N_Seq_Id : System.Address;
      Seq_Id   : System.Address;
      Logits   : System.Address;
   end record;
   pragma Convention (C, Llama_Batch);

   function Llama_Model_Default_Params return Llama_Model_Params;
   pragma Import (C, Llama_Model_Default_Params, "llama_model_default_params");

   function Llama_Context_Default_Params return Llama_Context_Params;
   pragma Import
     (C, Llama_Context_Default_Params, "llama_context_default_params");

   procedure Llama_Backend_Init;
   pragma Import (C, Llama_Backend_Init, "llama_backend_init");

   procedure Llama_Backend_Free;
   pragma Import (C, Llama_Backend_Free, "llama_backend_free");

   function Llama_Model_Load_From_File
     (Path_Model : chars_ptr; Params : Llama_Model_Params) return Llama_Model;
   pragma Import
     (C, Llama_Model_Load_From_File, "llama_model_load_from_file");

   procedure Llama_Model_Free (Model : Llama_Model);
   pragma Import (C, Llama_Model_Free, "llama_model_free");

   function Llama_Init_From_Model
     (Model : Llama_Model; Params : Llama_Context_Params) return Llama_Context;
   pragma Import (C, Llama_Init_From_Model, "llama_init_from_model");

   procedure Llama_Free (Context : Llama_Context);
   pragma Import (C, Llama_Free, "llama_free");

   function Llama_Batch_Init
     (N_Tokens : int; Embd : int; N_Seq_Max : int) return Llama_Batch;
   pragma Import (C, Llama_Batch_Init, "llama_batch_init");

   procedure Llama_Batch_Free (Batch : Llama_Batch);
   pragma Import (C, Llama_Batch_Free, "llama_batch_free");

   function Llama_Decode
     (Context : Llama_Context; Batch : Llama_Batch) return int;
   pragma Import (C, Llama_Decode, "llama_decode");

   function Llama_Get_Logits (Context : Llama_Context) return System.Address;
   pragma Import (C, Llama_Get_Logits, "llama_get_logits");

   procedure Llama_Set_Embeddings (Context : Llama_Context; Value : Boolean);
   pragma Import (C, Llama_Set_Embeddings, "llama_set_embeddings");

   function Llama_Get_Embeddings (Context : Llama_Context) return System.Address;
   pragma Import (C, Llama_Get_Embeddings, "llama_get_embeddings");

   procedure Llama_Set_N_Threads
     (Context : Llama_Context; N_Threads : int; N_Threads_Batch : int);
   pragma Import (C, Llama_Set_N_Threads, "llama_set_n_threads");

   function Llama_N_Vocab (Model : Llama_Model) return int;
   pragma Import (C, Llama_N_Vocab, "llama_n_vocab");

   function Llama_Model_Get_Vocab (Model : Llama_Model) return Llama_Vocab;
   pragma Import (C, Llama_Model_Get_Vocab, "llama_model_get_vocab");

   function Llama_Vocab_N_Tokens (Vocab : Llama_Vocab) return int;
   pragma Import (C, Llama_Vocab_N_Tokens, "llama_vocab_n_tokens");

   function Llama_Vocab_Is_Eog
     (Vocab : Llama_Vocab; Token : Llama_Token) return Boolean;
   pragma Import (C, Llama_Vocab_Is_Eog, "llama_vocab_is_eog");

   function Llama_Token_To_Piece
     (Vocab   : Llama_Vocab;
      Token   : Llama_Token;
      Buf     : System.Address;
      Length  : int;
      Lstrip  : int;
      Special : Boolean) return int;
   pragma Import (C, Llama_Token_To_Piece, "llama_token_to_piece");

   function Llama_Tokenize
     (Vocab        : Llama_Vocab;
      Text         : chars_ptr;
      Text_Len     : int;
      Tokens       : System.Address;
      N_Tokens_Max : int;
      Add_Special  : Boolean;
      Parse_Special : Boolean) return int;
   pragma Import (C, Llama_Tokenize, "llama_tokenize");

   function Llama_Detokenize
     (Vocab          : Llama_Vocab;
      Tokens         : System.Address;
      N_Tokens       : int;
      Text           : chars_ptr;
      Text_Len_Max   : int;
      Remove_Special : Boolean;
      Unparse_Special : Boolean) return int;
   pragma Import (C, Llama_Detokenize, "llama_detokenize");

   type Llama_Sampler_Chain_Params is record
      No_Perf : Boolean;
   end record;
   pragma Convention (C, Llama_Sampler_Chain_Params);

   --  Sampling API
   function Llama_Sampler_Chain_Default_Params return Llama_Sampler_Chain_Params;
   pragma Import
     (C, Llama_Sampler_Chain_Default_Params, "llama_sampler_chain_default_params");

   function Llama_Sampler_Chain_Init
     (Params : Llama_Sampler_Chain_Params) return Llama_Sampler;
   pragma Import (C, Llama_Sampler_Chain_Init, "llama_sampler_chain_init");

   procedure Llama_Sampler_Chain_Add (Chain : Llama_Sampler; Smpl : Llama_Sampler);
   pragma Import (C, Llama_Sampler_Chain_Add, "llama_sampler_chain_add");

   function Llama_Sampler_Init_Greedy return Llama_Sampler;
   pragma Import (C, Llama_Sampler_Init_Greedy, "llama_sampler_init_greedy");

   function Llama_Sampler_Init_Top_K (K : int) return Llama_Sampler;
   pragma Import (C, Llama_Sampler_Init_Top_K, "llama_sampler_init_top_k");

   function Llama_Sampler_Init_Top_P
     (P : Float; Min_Keep : size_t) return Llama_Sampler;
   pragma Import (C, Llama_Sampler_Init_Top_P, "llama_sampler_init_top_p");
function Llama_Sampler_Init_Temp (T : Float) return Llama_Sampler;
pragma Import (C, Llama_Sampler_Init_Temp, "llama_sampler_init_temp");

function Llama_Sampler_Init_Dist (Seed : unsigned) return Llama_Sampler;
pragma Import (C, Llama_Sampler_Init_Dist, "llama_sampler_init_dist");

function Llama_Sampler_Init_Penalties

     (Penalty_Last_N : int;
      Penalty_Repeat : Float;
      Penalty_Freq   : Float;
      Penalty_Present : Float) return Llama_Sampler;
   pragma Import
     (C, Llama_Sampler_Init_Penalties, "llama_sampler_init_penalties");

   function Llama_Sampler_Sample
     (Smpl : Llama_Sampler; Context : Llama_Context; Idx : int) return Llama_Token;
   pragma Import (C, Llama_Sampler_Sample, "llama_sampler_sample");

   procedure Llama_Sampler_Free (Smpl : Llama_Sampler);
   pragma Import (C, Llama_Sampler_Free, "llama_sampler_free");

   function Llama_Print_System_Info return chars_ptr;
   pragma Import (C, Llama_Print_System_Info, "llama_print_system_info");

end Llama_Interface;
