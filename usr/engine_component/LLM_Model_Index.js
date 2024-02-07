const availableImplementedLLMModelSpecificCategory = {
	// Usually the general_conversation have the most knowledge out them all
	general_conversation : {
		downloadLink: "https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b-v0.1.Q4_0.gguf?download=true", //change this later on
		filename: "generalLM_mist.bin",
		CategoryDescription: "Mixed Specialty General Conversation",
		ParamSize: "7B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 1
	},
	// Since mixtral exists we don't need the other model for now
	programming : {
		//downloadLink: "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_DeepSeekProgrammingModel.bin",
		CategoryDescription: "Programming Coding Focused Model Deepseek Chinese Model",
		ParamSize: "6.7B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 1
	},
	language_specific_indonesia : {
		//downloadLink: "https://huggingface.co/detakarang/sidrap-7b-v2-gguf/resolve/main/sidrap-7b-v2.q4_K_M.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_IndonesiaLanguage.bin",
		CategoryDescription: "Indonesian Language Conversation Maybe Translation or Something",
		ParamSize: "7B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 1
	},
	language_specific_japanese : {
		//downloadLink: "https://huggingface.co/TheBloke/japanese-stablelm-instruct-gamma-7B-GGUF/resolve/main/japanese-stablelm-instruct-gamma-7b.Q4_0.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_Japanese_Gamma_GGUF.bin",
		CategoryDescription: "General Japanese Conversation",
		ParamSize: "7B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 1
	},
	language_specific_english : {
		//downloadLink: "https://huggingface.co/maddes8cht/mosaicml-mpt-7b-gguf/resolve/main/mosaicml-mpt-7b-Q4_0.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_MosaicML_English.bin",
		CategoryDescription: "General Conversation",
		ParamSize: "7B",
		Quantization: "q4",
		Engine: "mpt",
		memAllocCutRatio: 1
	},
	language_specific_arabics :{
		//downloadLink: "https://www.youtube.com/watch?v=dQw4w9WgXcQ", //change this later on
		downloadLink:"",
		filename: "specLM_.bin",
		CategoryDescription: "General Conversation",
		ParamSize: "7B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 1
	},
	chemistry : {
		//downloadLink: "https://huggingface.co/TheBloke/Yi-34B-GiftedConvo-merged-GGUF/resolve/main/yi-34b-giftedconvo-merged.Q2_K.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_PhysicsandChemistry.bin",
		CategoryDescription: "General Conversation",
		ParamSize: "34B",
		Quantization: "q2_K",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.2
	},
	biology : {
		//downloadLink: "https://huggingface.co/TheBloke/medicine-LLM-13B-GGUF/resolve/main/medicine-llm-13b.Q4_0.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_BioMedicineLLM.bin",
		CategoryDescription: "BioMedicineLLM Conversation",
		ParamSize: "13B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.5
	},
	physics : {
		//downloadLink: "https://huggingface.co/TheBloke/Yi-34B-GiftedConvo-merged-GGUF/resolve/main/yi-34b-giftedconvo-merged.Q2_K.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_PhysicsandChemistry.bin",
		CategoryDescription: "General Conversation",
		ParamSize: "34B",
		Quantization: "q2_K",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.2
	},
	legal : {
		//downloadLink: "https://huggingface.co/TheBloke/law-LLM-GGUF/resolve/main/law-llm.Q4_0.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_LawLLM.bin",
		CategoryDescription: "Legal and Law Conversation",
		ParamSize: "7B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 1
	},
	medical_specific_science : {
		//downloadLink: "https://huggingface.co/TheBloke/medicine-LLM-13B-GGUF/resolve/main/medicine-llm-13b.Q4_0.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_BioMedicineLLM.bin",
		CategoryDescription: "Medical Specific Conversation",
		ParamSize: "13B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.4
	},
	//
	mathematics : {
		//downloadLink: "https://huggingface.co/TheBloke/llemma_7b-GGUF/resolve/main/llemma_7b.Q4_0.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_LLaMA_Math.bin",
		CategoryDescription: "Math LLM Operation",
		ParamSize: "7B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 1
	},
	
	financial : {
		//downloadLink: "https://huggingface.co/TheBloke/finance-LLM-GGUF/resolve/main/finance-llm.Q4_0.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_FinanceAdaptLLM.bin",
		CategoryDescription: "Financial Operation",
		ParamSize: "7B",
		Quantization: "q4",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 1
	},
	
	history : {
		//downloadLink: "https://huggingface.co/TheBloke/WizardLM-Uncensored-SuperCOT-StoryTelling-30B-GGUF/resolve/main/WizardLM-Uncensored-SuperCOT-Storytelling.Q2_K.gguf?download=true", //change this later on
		downloadLink:"",
		filename: "specLM_WizardLM_History_StoryTelling.bin",
		CategoryDescription: "History Conversation",
		ParamSize: "30B",
		Quantization: "q2_K",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.2
	},
};

module.exports = availableImplementedLLMModelSpecificCategory;