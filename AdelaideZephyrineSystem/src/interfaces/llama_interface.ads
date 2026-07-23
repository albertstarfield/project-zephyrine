pragma SPARK_Mode (Off);
-- c_binding: Llama.cpp C FFI
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

package Llama_Interface is

   type Llama_Model is new System.Address; -- FFI: System.Address required for C binding
   type Llama_Context is new System.Address; -- FFI: System.Address required for C binding
   type Llama_Vocab is new System.Address; -- FFI: System.Address required for C binding
   type Llama_Sampler is new System.Address; -- FFI: System.Address required for C binding

   type Llama_Token is new int;
   type Llama_Pos is new int;
   type Llama_Seq_Id is new int;

   Null_Model   : constant Llama_Model := Llama_Model (System.Null_Address);
   Null_Context : constant Llama_Context :=
     Llama_Context (System.Null_Address);

   GGML_TYPE_F32  : constant int := 0;
   GGML_TYPE_F16  : constant int := 1;
   GGML_TYPE_Q4_0 : constant int := 2;
   GGML_TYPE_Q4_1 : constant int := 3;

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
      Devices                    : System.Address; -- FFI: System.Address required for C binding
      Tensor_Buft_Overrides      : System.Address; -- FFI: System.Address required for C binding
      N_Gpu_Layers               : int;
      Split_Mode                 : int;
      Main_Gpu                   : int;
      Tensor_Split               : System.Address; -- FFI: System.Address required for C binding
      Progress_Callback          : System.Address; -- FFI: System.Address required for C binding
      Progress_Callback_User_Data : System.Address; -- FFI: System.Address required for C binding
      Kv_Overrides               : System.Address; -- FFI: System.Address required for C binding
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

   --  [VITAL-DO-NOT-REMOVE] Llama_Context_Params MUST match llama.h exactly.
   --  FFI struct layout mismatch causes silent corruption — fields after the
   --  missing one shift by 4 bytes (uint32_t size). The crash manifests as:
   --    ggml.c: GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT) failed
   --  because Type_K/Type_V end up at wrong offsets, writing garbage into
   --  the type enum fields. The C code reads invalid ggml_type values.
   --  CRASH LOG (2026-06-22, llama.cpp b9757):
   --    N_Outputs_Max was missing between N_Rs_Seq and N_Threads.
   --    This shifted EVERY subsequent field by 4 bytes:
   --    - Ada Type_K at offset 96, C type_k at offset 100
   --    - Ada Embeddings at offset 120, C embeddings at offset 124
   --    Result: KV cache creation reads garbage type → assertion failure
   --  FIX: Added N_Outputs_Max and Ctx_Other to match llama.h b9757 layout.
   --  When updating llama.cpp, ALWAYS diff llama_context_params in llama.h
   --  against this record and add/remove fields to match exactly.
   type Llama_Context_Params is record
      N_Ctx           : unsigned;
      N_Batch         : unsigned;
      N_Ubatch        : unsigned;
      N_Seq_Max       : unsigned;
      N_Rs_Seq        : unsigned;
      N_Outputs_Max   : unsigned;     -- [VITAL-DO-NOT-REMOVE] See comment block above.
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

      Cb_Eval           : System.Address; -- FFI: System.Address required for C binding
      Cb_Eval_User_Data : System.Address; -- FFI: System.Address required for C binding

      Type_K : int;
      Type_V : int;

      Abort_Callback      : System.Address; -- FFI: System.Address required for C binding
      Abort_Callback_Data : System.Address; -- FFI: System.Address required for C binding

      Embeddings  : Boolean;
      Offload_Kqv : Boolean;
      No_Perf     : Boolean;
      Op_Offload  : Boolean;
      Swa_Full    : Boolean;
      Kv_Unified  : Boolean;

      Samplers    : System.Address; -- FFI: System.Address required for C binding
      N_Samplers  : size_t;
      --  [VITAL-DO-NOT-REMOVE] Ctx_Other is the last field in llama.h's
      --  llama_context_params struct (b9757). Without it, the Ada struct
      --  is smaller than the C struct. While this doesn't shift other fields,
      --  it means llama_context_default_params() fills more bytes than we
      --  have storage for, which could corrupt memory if the struct grows.
      Ctx_Other   : System.Address; -- FFI: System.Address required for C binding
   end record;
   pragma Convention (C, Llama_Context_Params);

   type Llama_Batch is record
      N_Tokens : int;
      Token    : System.Address; -- FFI: System.Address required for C binding
      Embd     : System.Address; -- FFI: System.Address required for C binding
      Pos      : System.Address; -- FFI: System.Address required for C binding
      N_Seq_Id : System.Address; -- FFI: System.Address required for C binding
      Seq_Id   : System.Address; -- FFI: System.Address required for C binding
      Logits   : System.Address; -- FFI: System.Address required for C binding
   end record;
   pragma Convention (C, Llama_Batch);

   --  Returns the default model loading parameters for llama.cpp.
   function Llama_Model_Default_Params return Llama_Model_Params;
   pragma Import (C, Llama_Model_Default_Params, "llama_model_default_params");

   --  Returns the default context parameters for llama.cpp.
   function Llama_Context_Default_Params return Llama_Context_Params;
   pragma Import
     (C, Llama_Context_Default_Params, "llama_context_default_params");

   --  Initializes the llama.cpp backend. Must be called once before any other llama functions.
   procedure Llama_Backend_Init;
   pragma Import (C, Llama_Backend_Init, "llama_backend_init");

   --  Frees the llama.cpp backend resources. Call once at shutdown.
   procedure Llama_Backend_Free;
   pragma Import (C, Llama_Backend_Free, "llama_backend_free");

   --  Loads a GGUF model from the specified file path with the given parameters.
   --  Returns Null_Model on failure.
   function Llama_Model_Load_From_File
     (Path_Model : chars_ptr; Params : Llama_Model_Params) return Llama_Model;
   pragma Import
     (C, Llama_Model_Load_From_File, "llama_model_load_from_file_safe");

   --  Frees a previously loaded model and releases its resources.
   procedure Llama_Model_Free (Model : Llama_Model);
   pragma Import (C, Llama_Model_Free, "llama_model_free");

    --  Creates a new inference context from a loaded model with the given parameters.
    --  Returns a context handle that can be used for tokenization and decoding.
     function Llama_Init_From_Model
      (Model : Llama_Model; Params : Llama_Context_Params) return Llama_Context;
     pragma Import (C, Llama_Init_From_Model, "llama_init_from_model");

    --  Safe variant of Llama_Init_From_Model with C++ exception safety at the FFI boundary.
     function Llama_Init_From_Model_Safe
      (Model : Llama_Model; Params : Llama_Context_Params) return Llama_Context;
    pragma Import (C, Llama_Init_From_Model_Safe, "llama_init_from_model_safe");


   --  Frees a previously created context and releases its resources.
   procedure Llama_Free (Context : Llama_Context);
   pragma Import (C, Llama_Free, "llama_free");

   procedure Llama_Memory_Clear (Mem : System.Address; Data : Boolean); -- FFI: System.Address required for C binding
   pragma Import (C, Llama_Memory_Clear, "llama_memory_clear");

   --  Removes a sequence from the KV cache memory, clearing all tokens in that sequence.
   --  Returns True on success.
   function Llama_Memory_Seq_Rm
     (Mem : System.Address; Seq_Id : int; P0 : int; P1 : int) return Boolean; -- FFI: System.Address required for C binding
   pragma Import (C, Llama_Memory_Seq_Rm, "llama_memory_seq_rm");

   function Llama_Get_Memory (Context : Llama_Context) return System.Address; -- FFI: System.Address required for C binding
   pragma Import (C, Llama_Get_Memory, "llama_get_memory");

   --  Returns the number of tokens in the context's KV cache.
   function Llama_N_Ctx (Context : Llama_Context) return Interfaces.C.unsigned;
   pragma Import (C, Llama_N_Ctx, "llama_n_ctx");

   --  Saves the full context state to a file, including the KV cache and optional tokens.
   --  Returns True on success.
   function Llama_State_Save_File
     (Context : Llama_Context; Path : chars_ptr; Tokens : System.Address; N_Tokens : size_t) return Boolean; -- FFI: System.Address required for C binding
   pragma Import (C, Llama_State_Save_File, "llama_state_save_file");

   --  Loads a previously saved context state from a file into the given context.
   --  Returns True on success.
   function Llama_State_Load_File
     (Context : Llama_Context; Path : chars_ptr; Tokens : System.Address; N_Tokens : size_t; N_Tokens_Out : access size_t) return Boolean; -- FFI: System.Address required for C binding
   pragma Import (C, Llama_State_Load_File, "llama_state_load_file");

   --  Returns the byte size needed to store the full context state in memory.
   function Llama_State_Get_Size
     (Context : Llama_Context) return size_t;
   pragma Import (C, Llama_State_Get_Size, "llama_state_get_size");

   --  Copies the context state into the destination buffer. Returns bytes written.
   function Llama_State_Get_Data
     (Context : Llama_Context; Dst : System.Address; Size : size_t) return size_t; -- FFI: System.Address required for C binding
   pragma Import (C, Llama_State_Get_Data, "llama_state_get_data");

   --  Restores the context state from the source buffer. Returns bytes consumed.
   function Llama_State_Set_Data
     (Context : Llama_Context; Src : System.Address; Size : size_t) return size_t; -- FFI: System.Address required for C binding
   pragma Import (C, Llama_State_Set_Data, "llama_state_set_data");

   --  Allocates and initializes a batch structure for the given token/embedding/sequence counts.
   function Llama_Batch_Init
     (N_Tokens : int; Embd : int; N_Seq_Max : int) return Llama_Batch;
   pragma Import (C, Llama_Batch_Init, "llama_batch_init");

   --  Safely appends a token to a batch at the given position and sequence, optionally
   --  requesting logits for this token.
   procedure Llama_Batch_Add_Safe
     (Batch : System.Address; Token : Llama_Token; Pos : int; Seq_Id : int; Logits : Boolean); -- FFI: System.Address required for C binding
   pragma Import (C, Llama_Batch_Add_Safe, "llama_batch_add_safe");

   procedure Llama_Batch_Clear_Safe (Batch : System.Address); -- FFI: System.Address required for C binding
   pragma Import (C, Llama_Batch_Clear_Safe, "llama_batch_clear_safe");

   --  Frees a previously allocated batch and its internal buffers.
   procedure Llama_Batch_Free (Batch : Llama_Batch);
   pragma Import (C, Llama_Batch_Free, "llama_batch_free");

   --  Decodes the batch through the context, computing forward pass for all tokens.
   --  Returns 0 on success, negative on failure.
   function Llama_Decode
     (Context : Llama_Context; Batch : Llama_Batch) return int;
   pragma Import (C, Llama_Decode, "llama_decode");

   function Llama_Get_Logits (Context : Llama_Context) return System.Address; -- FFI: System.Address required for C binding
   pragma Import (C, Llama_Get_Logits, "llama_get_logits");

   --  Enables or disables embedding extraction mode in the context.
   procedure Llama_Set_Embeddings (Context : Llama_Context; Value : Interfaces.C.int);
   pragma Import (C, Llama_Set_Embeddings, "llama_set_embeddings");

   function Llama_Get_Embeddings (Context : Llama_Context) return System.Address; -- FFI: System.Address required for C binding
   pragma Import (C, Llama_Get_Embeddings, "llama_get_embeddings");

   --  Sets the number of threads used for inference and batch processing.
   procedure Llama_Set_N_Threads
     (Context : Llama_Context; N_Threads : int; N_Threads_Batch : int);
   pragma Import (C, Llama_Set_N_Threads, "llama_set_n_threads");

   --  Returns the vocabulary size of the loaded model.
   function Llama_N_Vocab (Model : Llama_Model) return int;
   pragma Import (C, Llama_N_Vocab, "llama_n_vocab");

   --  Retrieves the vocabulary handle from a loaded model for tokenization operations.
   function Llama_Model_Get_Vocab (Model : Llama_Model) return Llama_Vocab;
   pragma Import (C, Llama_Model_Get_Vocab, "llama_model_get_vocab");

   --  Returns the total number of tokens in the vocabulary.
   function Llama_Vocab_N_Tokens (Vocab : Llama_Vocab) return int;
   pragma Import (C, Llama_Vocab_N_Tokens, "llama_vocab_n_tokens");

   --  Returns True if the given token is an end-of-generation (EOG) token.
   function Llama_Vocab_Is_Eog
     (Vocab : Llama_Vocab; Token : Llama_Token) return Boolean;
   pragma Import (C, Llama_Vocab_Is_Eog, "llama_vocab_is_eog");

   --  Converts a token ID to its text representation, writing up to Length bytes
   --  into the provided buffer. Returns the number of bytes written.
   function Llama_Token_To_Piece
     (Vocab   : Llama_Vocab;
      Token   : Llama_Token;
      Buf     : System.Address; -- FFI: System.Address required for C binding
      Length  : int;
      Lstrip  : int;
      Special : Boolean) return int;
   pragma Import (C, Llama_Token_To_Piece, "llama_token_to_piece");

   --  Tokenizes the input text string, writing token IDs into the provided buffer.
   --  Returns the number of tokens produced.
   function Llama_Tokenize
     (Vocab        : Llama_Vocab;
      Text         : chars_ptr;
      Text_Len     : int;
      Tokens       : System.Address; -- FFI: System.Address required for C binding
      N_Tokens_Max : int;
      Add_Special  : Boolean;
      Parse_Special : Boolean) return int;
   pragma Import (C, Llama_Tokenize, "llama_tokenize");

   --  Converts an array of token IDs back into text, writing up to Text_Len_Max
   --  bytes. Returns the number of bytes written.
   function Llama_Detokenize
     (Vocab          : Llama_Vocab;
      Tokens         : System.Address; -- FFI: System.Address required for C binding
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

   --  Initializes a new sampler chain with the given parameters.
   function Llama_Sampler_Chain_Init
     (Params : Llama_Sampler_Chain_Params) return Llama_Sampler;
   pragma Import (C, Llama_Sampler_Chain_Init, "llama_sampler_chain_init");

   --  Appends a sampler to the end of a sampler chain.
   procedure Llama_Sampler_Chain_Add (Chain : Llama_Sampler; Smpl : Llama_Sampler);
   pragma Import (C, Llama_Sampler_Chain_Add, "llama_sampler_chain_add");

   --  Creates a greedy sampler that always selects the highest-probability token.
   function Llama_Sampler_Init_Greedy return Llama_Sampler;
   pragma Import (C, Llama_Sampler_Init_Greedy, "llama_sampler_init_greedy");

   --  Creates a Top-K sampler that considers only the K most probable tokens.
   function Llama_Sampler_Init_Top_K (K : int) return Llama_Sampler;
   pragma Import (C, Llama_Sampler_Init_Top_K, "llama_sampler_init_top_k");

   --  Creates a Top-P (nucleus) sampler that considers tokens within cumulative
   --  probability P, keeping at least Min_Keep tokens.
   function Llama_Sampler_Init_Top_P
     (P : Float; Min_Keep : size_t) return Llama_Sampler;
   pragma Import (C, Llama_Sampler_Init_Top_P, "llama_sampler_init_top_p");
   --  Creates a temperature sampler that adjusts token probabilities by the given temperature T.
function Llama_Sampler_Init_Temp (T : Float) return Llama_Sampler;
pragma Import (C, Llama_Sampler_Init_Temp, "llama_sampler_init_temp");

   --  Creates a distribution sampler using the given random seed for reproducibility.
function Llama_Sampler_Init_Dist (Seed : unsigned) return Llama_Sampler;
pragma Import (C, Llama_Sampler_Init_Dist, "llama_sampler_init_dist");

   --  Creates a penalties sampler that applies repeat/frequency/presence penalties
   --  to discourage token repetition.
function Llama_Sampler_Init_Penalties

     (Penalty_Last_N : int;
      Penalty_Repeat : Float;
      Penalty_Freq   : Float;
      Penalty_Present : Float) return Llama_Sampler;
   pragma Import
     (C, Llama_Sampler_Init_Penalties, "llama_sampler_init_penalties");

   --  Samples a single token from the context using the given sampler at the specified position.
   function Llama_Sampler_Sample
     (Smpl : Llama_Sampler; Context : Llama_Context; Idx : int) return Llama_Token;
   pragma Import (C, Llama_Sampler_Sample, "llama_sampler_sample");

   --  Frees a previously created sampler and its resources.
   procedure Llama_Sampler_Free (Smpl : Llama_Sampler);
   pragma Import (C, Llama_Sampler_Free, "llama_sampler_free");

   --  Returns a C string containing system information about the llama.cpp build
   --  and available backends.
   function Llama_Print_System_Info return chars_ptr;
   pragma Import (C, Llama_Print_System_Info, "llama_print_system_info");

   --  ===== GPU MEMORY QUERY =====
   --  Queries GPU device memory (free/total) through ggml backend.
   --  Works for ALL backends: Metal (Apple), CUDA (NVIDIA), OneAPI/SYCL (Intel),
   --  Vulkan (cross-platform), ROCm (AMD), NNA (Qualcomm).
   --  For CPU-only: returns 0,0 (inapplicable).
   procedure GPU_Memory_Query
     (Free_Bytes  : out Interfaces.C.size_t;
      Total_Bytes : out Interfaces.C.size_t);
   pragma Import (C, GPU_Memory_Query, "gpu_memory_query");

   --  ===== CPU MEMORY QUERY =====
   --  Returns free and total CPU memory in bytes.
   --  Uses macOS host_statistics64 for free memory and sysctl for total.
   procedure CPU_Memory_Query
     (Free_Bytes  : out Interfaces.C.size_t;
      Total_Bytes : out Interfaces.C.size_t);
   pragma Import (C, CPU_Memory_Query, "cpu_memory_query");

   --  Returns the number of available CPU threads (physical + hyperthreaded).
   --  Uses sysctl HW_NCPU on macOS.
   function CPU_Thread_Count return Interfaces.C.unsigned;
   pragma Import (C, CPU_Thread_Count, "cpu_thread_count");

   --  ===== RERANKING API =====
   --  When Pooling_Type = LLAMA_POOLING_TYPE_RANK (4), llama.cpp attaches
   --  a classification head to the graph for reranking models like
   --  Qwen3-Reranker-0.6B. After decode, llama_get_embeddings_seq returns
   --  float[n_cls_out] with relevance scores per sequence.
   --
   --  Usage:
   --    1. Load reranker model with Pooling_Type => 4 (RANK)
   --    2. Tokenize "query\t\tdocument" pairs
   --    3. llama_decode each pair
   --    4. llama_get_embeddings_seq returns float[n_cls_out] (score)
   --
   function Llama_Get_Embeddings_Seq
     (Context : Llama_Context;
      Seq_Id  : Interfaces.C.int) return System.Address; -- FFI: System.Address required for C binding
   pragma Import (C, Llama_Get_Embeddings_Seq, "llama_get_embeddings_seq");

   --  Returns number of classifier outputs for reranking models.
   --  Undefined for non-classifier models.
   function Llama_Model_N_Cls_Out
     (Model : Llama_Model) return Interfaces.C.unsigned;
   pragma Import (C, Llama_Model_N_Cls_Out, "llama_model_n_cls_out");

   --  Returns the pooling type of the context.
   function Llama_Pooling_Type
     (Context : Llama_Context) return Interfaces.C.int;
   pragma Import (C, Llama_Pooling_Type, "llama_pooling_type");

end Llama_Interface;
