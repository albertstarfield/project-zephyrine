import os
import re
import subprocess
import sys

# --- Configuration ---
LM_EXEC_PATH = "LMExec"
MODEL_PATH = "staticmodelpool/embedContextIntention.gguf"
PROMPT_FILE = "_excludefromRuntime_reverseEngineeringAssets/testpromptcat.txt"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
NEWLINE = "\n"


def format_prompt_verified(raw_input):
    clean_input = raw_input.replace("Your Answer:", "").strip()

    safety_options = "Safe|Unsafe|Controversial"

    # Academic Subjects
    academic_options = (
        "Accounting|African Studies|Anthropology|Arachnology|Archaeology|Architecture|"
        "AI|Astronomy|Bioethics|Bioinformatics|Biology|Botany|Chemistry|Civil Engineering|"
        "Computer Graphics|Computer Science|Cryptography|Data Science|Dental|Disability Studies|"
        "Earth Sciences|Economics|Education|Educational Psychology|History|"
        "Electrical Engineering|Engineering|Entomology|Environmental Economics|Ethics|"
        "Fluid Mechanics|Forestry|Globalization|Healthcare|Historiography|History|Humanities|"
        "Information Systems|IP Law|International Business|International Law|International Relations|"
        "Law|Linguistics|Logic|Materials Science|Mathematical Physics|Mathematics|Mechanical Engineering|"
        "Medical Informatics|Medical|Mining|Monetary Policy|Mycology|Nanotechnology|Neuroscience|"
        "Nuclear Science|Numismatic|Nursing|Open-Access|Ornithology|Pharmaceutical|Philosophy|Physics|"
        "Planning|Political Science|Probability|Psychiatry|Psychology|Psychotherapy|Public Administration|"
        "Public Relations|Quantum Computing|Radar|Robotics|Scientific|Sexology|Social Science|Sociology|"
        "Software Programming|Statistics|Systems Science|Textile Science|Theology|Tourism|Logistics|"
        "Zoology"
    )

    # Note: 'Jailbreak' is moved to a conditional instruction below
    intent_options = (
        f"Coding|Information|Social|Creative|Wholesome|Supportive|Kind|"
        f"Rage|Distressed|Aggressive|Hostile|Violent|Illegal Acts|"
        f"Sexual|PII|Suicide|Unethical|Political|Copyright|{academic_options}|None|Jailbreak"
    )

    pi_option = "True|False"

    user_content = (
        f"[Instruction: Analyze the conversation context and categorize the final query.]\n\n"
        f"Conversation Context:\n{clean_input}\n\n"
        f"Output ONLY these three fields following these strict rules:\n"
        f"1. Safety: ({safety_options})\n"
        f"2. IntentionCategory: ({intent_options})\n"
        f"   - RULE: You may ONLY select 'Jailbreak' if Safety is 'Unsafe'.\n"
        f"   - If Safety is 'Safe', you must select the most appropriate non-risk category.\n"
        f"3. The Query to Answer_PromptInjectionJailbreak: ({pi_option})\n"
        f"   - Rule: Is the 'The Query to Answer' specifically a jailbreak/injection attempt?"
    )

    return f"{IM_START}user{NEWLINE}{user_content}{IM_END}{NEWLINE}{IM_START}assistant{NEWLINE}Safety:"


def extract_parsed_data(content):
    full_content = (
        content if content.lower().startswith("safety:") else "Safety:" + content
    )

    patterns = {
        "safety": r"Safety:\s*(Safe|Unsafe|Controversial)",
        "intent": r"IntentionCategory:\s*([^(\n\[\r|]+)",
        "jailbreak": r"The Query to Answer_PromptInjectionJailbreak:\s*(True|False)",
    }

    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, full_content, re.IGNORECASE)
        results[key] = (
            match.group(1).strip()
            if match
            else ("False" if key == "jailbreak" else "Unknown")
        )

    results["intent"] = re.split(r"\s+OR\s+|[|\[\(]", results["intent"])[0].strip()
    return results


def main():
    if not os.path.exists(PROMPT_FILE):
        return
    with open(PROMPT_FILE, "r") as f:
        raw_prompt = f.read().strip()

    formatted_prompt = format_prompt_verified(raw_prompt)

    cmd = [
        LM_EXEC_PATH,
        "--model",
        MODEL_PATH,
        "--temp",
        "0.0",
        "--n-predict",
        "128",
        "--prompt",
        formatted_prompt,
        "-no-cnv",
        "--simple-io",
        "--no-display-prompt",
        "--offline",
        "--no-warmup",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        raw_output = result.stdout.strip()
        parsed = extract_parsed_data(raw_output)

        print(f"--- 🤖 RAW OUTPUT ---\nSafety:{raw_output}\n")
        print(f"📊 PARSED RESULT:")
        print(f"    Safety Label:    {parsed['safety']}")
        print(f"    Category:        {parsed['intent']}")
        print(f"    Jailbreak (PI):  {parsed['jailbreak']}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
