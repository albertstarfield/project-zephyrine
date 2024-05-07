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
		downloadLink: "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_0.gguf?download=true", //change this later on
		filename: "default_generalLM_mist.bin",
		CategoryDescription: "Standard General Conversation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "system \n",
		instructionPrompt: "user \n",
		responsePrompt: "assistant \n",
		memAllocCutRatio: 1, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	// Since mixtral exists we don't need the other model, for now. However later on we'll going to accept willy model later on
	// mixtral can't run properly on 16GB of vram
	// going back to the old version of Zephy
	general_conversation_llama_3 : {
		downloadLink: "https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf?download=true", //change this later on
		filename: "generalLM_llama3Base.bin",
		CategoryDescription: "Big Context Window Support using Latest LLaMa3 Developed by meta for big interaction window",
		ParamSize: "8B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 1, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
		programming : {
		downloadLink: "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf?download=true", //change this later on
		filename: "specLM_DeepSeekProgrammingModel.bin",
		CategoryDescription: "Programming Coding Focused Model Deepseek Chinese Model",
		ParamSize: "6.7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	language_specific_indonesia : {
		downloadLink: "https://huggingface.co/janhq/komodo-7b-chat-GGUF/resolve/main/komodo-7b-chat.Q3_K_M.gguf?download=true", //change this later on
		filename: "specLM_IndonesiaLanguage.bin",
		CategoryDescription: "Indonesian Multi cultural Komodo7b Conversation Language Model",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	language_specific_japanese : {
		downloadLink: "https://huggingface.co/TheBloke/japanese-stablelm-instruct-gamma-7B-GGUF/resolve/main/japanese-stablelm-instruct-gamma-7b.Q4_0.gguf?download=true", //change this later on
		filename: "specLM_Japanese_Gamma_GGUF.bin",
		CategoryDescription: "General Japanese Conversation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	language_specific_english : {
		downloadLink: "https://huggingface.co/maddes8cht/mosaicml-mpt-7b-gguf/resolve/main/mosaicml-mpt-7b-Q4_0.gguf?download=true", //change this later on
		filename: "specLM_MosaicML_English.bin",
		CategoryDescription: "General Conversation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "mpt",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	language_specific_arabics :{
		downloadLink: "https://www.youtube.com/watch?v=dQw4w9WgXcQ", //change this later on
		filename: "specLM_.bin",
		CategoryDescription: "General Conversation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 1 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	chemistry : {
		downloadLink: "https://huggingface.co/TheBloke/Yi-34B-GiftedConvo-merged-GGUF/resolve/main/yi-34b-giftedconvo-merged.Q2_K.gguf?download=true", //change this later on
		filename: "specLM_PhysicsandChemistry.bin",
		CategoryDescription: "Focuses on Physics and Chemistry",
		ParamSize: "34B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.1, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 1 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	biology : {
		downloadLink: "https://huggingface.co/TheBloke/medicine-LLM-13B-GGUF/resolve/main/medicine-llm-13b.Q4_0.gguf?download=true", //change this later on
		filename: "specLM_BioMedicineLLM.bin",
		CategoryDescription: "Medicine or Medical Conversation but it's not to replace Doctors for initial diagnosis just for recommendation",
		ParamSize: "13B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	physics : {
		downloadLink: "https://huggingface.co/TheBloke/Yi-34B-GiftedConvo-merged-GGUF/resolve/main/yi-34b-giftedconvo-merged.Q2_K.gguf?download=true", //change this later on
		filename: "specLM_PhysicsandChemistry.bin",
		CategoryDescription: "Focuses on Physics and Chemistry",
		ParamSize: "34B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.1, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 1 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	legal : {
		downloadLink: "https://huggingface.co/TheBloke/law-LLM-GGUF/resolve/main/law-llm.Q4_0.gguf?download=true", //change this later on
		filename: "specLM_LawLLM.bin",
		CategoryDescription: "Legal and Law Conversation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	medical_specific_science : {
		downloadLink: "https://huggingface.co/TheBloke/medicine-LLM-13B-GGUF/resolve/main/medicine-llm-13b.Q4_0.gguf?download=true", //change this later on
		filename: "specLM_BioMedicineLLM.bin",
		CategoryDescription: "Medical Specific Conversation",
		ParamSize: "13B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.4, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	//
	mathematics : {
		downloadLink: "https://huggingface.co/TheBloke/llemma_7b-GGUF/resolve/main/llemma_7b.Q4_0.gguf?download=true", //change this later on
		filename: "specLM_LLaMA_Math.bin",
		CategoryDescription: "Mathematics Might be with Latex LLM Operation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	
	financial : {
		downloadLink: "https://huggingface.co/TheBloke/finance-LLM-GGUF/resolve/main/finance-llm.Q4_0.gguf?download=true", //change this later on
		filename: "specLM_FinanceAdaptLLM.bin",
		CategoryDescription: "Financial advise Interaction",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	history : {
		downloadLink: "https://huggingface.co/TheBloke/WizardLM-Uncensored-SuperCOT-StoryTelling-30B-GGUF/resolve/main/WizardLM-Uncensored-SuperCOT-Storytelling.Q2_K.gguf?download=true", //change this later on
		filename: "specLM_WizardLM_History_StoryTelling.bin",
		CategoryDescription: "History Conversation",
		ParamSize: "30B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 1 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	//Gemma
	googleGemma : {
		downloadLink: "", //change this later on, Gemma direct download link doesn't exist yet or blocked that you have to accept gemma license to be able to download it
		filename: "generalLM_gemma_noncontinous.bin",
		CategoryDescription: "Google General Conversation support",
		ParamSize: "7B",
		Quantization: "q8_0", // 8-bit quantization sfp
		Engine: "gemma",
		systemPrompt: "",
		instructionPrompt: "### Instruction: \n",
		responsePrompt: "### Response: \n",
		memAllocCutRatio: 1, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
    // Vision LLM Model Processed
	VisionLLM : {
		downloadLink: "https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-Q4_K.gguf?download=true", //change this later on
		filename: "specLM_Vision_LLaVa.bin",
		CategoryDescription: "LLM That can describe with image or framebuffer",
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
