const availableImplementedLLMModelSpecificCategory = {
	// Usually the general_conversation have the most knowledge out them all
	general_conversation : {
		downloadLink: "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_0.gguf?download=true", //change this later on
		filename: "generalLM_mist.bin",
		CategoryDescription: "Std General Conversation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 1, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	// Since mixtral exists we don't need the other model, for now. However later on we'll going to accept willy model later on
	// mixtral can't run properly on 16GB of vram
	// going back to the old version of Zephy
		programming : {
		downloadLink: "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf?download=true", //change this later on
		filename: "specLM_DeepSeekProgrammingModel.bin",
		CategoryDescription: "Programming Coding Focused Model Deepseek Chinese Model",
		ParamSize: "6.7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	language_specific_indonesia : {
		downloadLink: "https://huggingface.co/detakarang/sidrap-7b-v2-gguf/resolve/main/sidrap-7b-v2.q4_K_M.gguf?download=true", //change this later on
		filename: "specLM_IndonesiaLanguage.bin",
		CategoryDescription: "Indonesian Language Conversation Maybe Translation or Something",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
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
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 1 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	chemistry : {
		downloadLink: "https://huggingface.co/TheBloke/Yi-34B-GiftedConvo-merged-GGUF/resolve/main/yi-34b-giftedconvo-merged.Q2_K.gguf?download=true", //change this later on
		filename: "specLM_PhysicsandChemistry.bin",
		CategoryDescription: "General Conversation",
		ParamSize: "34B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.1, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 1 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	biology : {
		downloadLink: "https://huggingface.co/TheBloke/medicine-LLM-13B-GGUF/resolve/main/medicine-llm-13b.Q4_0.gguf?download=true", //change this later on
		filename: "specLM_BioMedicineLLM.bin",
		CategoryDescription: "BioMedicineLLM Conversation",
		ParamSize: "13B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	physics : {
		downloadLink: "https://huggingface.co/TheBloke/Yi-34B-GiftedConvo-merged-GGUF/resolve/main/yi-34b-giftedconvo-merged.Q2_K.gguf?download=true", //change this later on
		filename: "specLM_PhysicsandChemistry.bin",
		CategoryDescription: "General Conversation",
		ParamSize: "34B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
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
		memAllocCutRatio: 0.4, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	//
	mathematics : {
		downloadLink: "https://huggingface.co/TheBloke/llemma_7b-GGUF/resolve/main/llemma_7b.Q4_0.gguf?download=true", //change this later on
		filename: "specLM_LLaMA_Math.bin",
		CategoryDescription: "Math LLM Operation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	
	financial : {
		downloadLink: "https://huggingface.co/TheBloke/finance-LLM-GGUF/resolve/main/finance-llm.Q4_0.gguf?download=true", //change this later on
		filename: "specLM_FinanceAdaptLLM.bin",
		CategoryDescription: "Financial Operation",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
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
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 1 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	//Gemma
	googleGemma : {
		downloadLink: "", //change this later on, Gemma direct download link doesn't exist yet or blocked that you have to accept gemma license to be able to download it
		filename: "generalLM_gemma_noncontinous.bin",
		CategoryDescription: "Google General Conversation support",
		ParamSize: "7B",
		Quantization: "q8_0", // 8-bit quantization
		Engine: "gemma",
		memAllocCutRatio: 1, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
    // Vision LLM Model Processed
	VisionLLM : {
		downloadLink: "https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-Q4_K.gguf?download=true", //change this later on
		filename: "specLM_Vision_LLaVa.bin",
		CategoryDescription: "LLM That can describe with image",
		ParamSize: "7B",
		Quantization: "q4_0",
		Engine: "LLaMa2gguf",
		memAllocCutRatio: 0.6, // The ratio for offloading which determines the --ngl layer from the setting for instance {settings}*{memAllocCutRatio} (example 512*0.4= number of layers offloaded ngl for cache)
		diskEnforceWheelspin: 0 //T****e the disk to oblivion and offload from ram (this is recommended on solid state PCIE6 10GB/s) and may leave you at the end of the day 100TB/day of read and may leave the storage controller operating at 100C constantly. This is part of my motivation if you don't force yourself, you'll be irrelevant. This will set -ngld or the offload draft layer to 9999 -ngl 1
	},
	// Speech 
};

module.exports = availableImplementedLLMModelSpecificCategory;
