const availableImplementedLLMModelSpecificCategory = {
	// Usually the general_conversation have the most knowledge out them all
	/*
	<|im_start|>system
	{system_message}<|im_end|>
	<|im_start|>user
	{prompt}<|im_end|>
	<|im_start|>assistant

	<|im_xxx|> is token we can just ignore it
	*/
	// General_conversation is a model that is universally acceptable, that know at least the generic general concept, then later on we can retrain it with LLM Base-model way of unsupervised training and supervised training from the interaction (we call the process Re-integration)
	general_conversation : {
		downloadLink: "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q2_K.gguf?download=true", //change this later on
		filename: "raw_smallLanguageModel.bin",
		filename_selfRemorphedRetraining: "evolved_smallLanguageModel.bin",
		CategoryDescription: "Main General Conversation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "{system} \n",
		instructionPrompt: "{user} \n",
		responsePrompt: "{assistant} \n",
		memAllocCutRatio: 1, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	// Since mixtral exists we don't need the other model, for now. However later on we'll going to accept willy model later on
    // Vision LLM Model Processed
	VisionLLM : {
		downloadLink: "https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-Q4_K.gguf?download=true", //change this later on
		filename: "specLM_Vision_LLaVa.bin",
		CategoryDescription: "LLM That can describe image or framebuffer",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	// Speech 
};


// Logs On the Model Tuning
// 146B Parameter LLM Total Model For this Adelaide Paradigm 0.1.3 

module.exports = availableImplementedLLMModelSpecificCategory;
