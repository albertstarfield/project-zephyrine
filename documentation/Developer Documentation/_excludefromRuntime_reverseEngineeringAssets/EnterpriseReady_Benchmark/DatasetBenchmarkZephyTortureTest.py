import time
import random
import os
import json
from openai import OpenAI
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm
import re
import numpy as np
from colorama import Fore, Style, init as colorama_init

# --- Hugging Face Cache Configuration ---
CACHE_DIR_NAME = "hf_benchmark_dataset_cache"
current_working_directory = os.getcwd()
hf_cache_path = os.path.join(current_working_directory, CACHE_DIR_NAME)
os.environ['HF_DATASETS_CACHE'] = hf_cache_path
os.environ['HF_HOME'] = hf_cache_path
os.makedirs(hf_cache_path, exist_ok=True)

# --- Initialize Colorama ---
colorama_init(autoreset=True)

# --- Color Definitions ---
C_INFO = Fore.CYAN
C_SUCCESS = Fore.GREEN
C_WARNING = Fore.YELLOW
C_ERROR = Fore.RED
C_BOLD = Style.BRIGHT
C_RESET = Style.RESET_ALL
C_SUBJECT = Fore.MAGENTA
C_METRIC = Fore.BLUE
C_DETAIL = Fore.LIGHTBLACK_EX
C_AI_RESPONSE = Fore.LIGHTMAGENTA_EX

# --- Configuration ---
LOCAL_API_BASE_URL = "http://localhost:11434/v1"
LOCAL_API_KEY = "NA"
MODEL_NAME = "Amaryllis-AdelaidexAlbert-MetacognitionArtificialQuellia"

# MMLU Configuration - NOW RUNNING ALL SUBJECTS, 100 QUESTIONS EACH
# WARNING: This will take a very long time!
try:
    ALL_MMLU_SUBJECTS = get_dataset_config_names("cais/mmlu")
    MMLU_SUBJECTS = ALL_MMLU_SUBJECTS  # Use all subjects
except Exception as e:
    print(f"{C_ERROR}{C_BOLD}FATAL ERROR: Could not fetch MMLU subject names from Hugging Face: {e}{C_RESET}")
    print(
        f"{C_WARNING}Defaulting to a small subset of MMLU subjects. Please check your internet connection or HF Hub status.{C_RESET}")
    MMLU_SUBJECTS = ["abstract_algebra", "computer_security", "us_foreign_policy"]  # Fallback

QUESTIONS_PER_SUBJECT_LIMIT = 100  # Run 100 questions per subject. Set to None for all available (usually <100 for test split).
RANDOM_SEED = 42

# Enterprise Worthiness Thresholds
ACCURACY_WORTHINESS_THRESHOLD_PERCENT = 80.0
LATENCY_WORTHINESS_THRESHOLD_MS = 3000.0

# --- Debugging & Logging ---
PRINT_PER_QUESTION_DETAILS_DURING_RUN = False
PRINT_FULL_API_LOG_AT_END = False  # Recommended to keep False for such a large run to save console space/log file size
FEEDBACK_TO_AI_ENABLED = True

# --- Global list for API interaction logging ---
api_interaction_log = []

# --- Initialize OpenAI Client ---
client = OpenAI(
    base_url=LOCAL_API_BASE_URL,
    api_key=LOCAL_API_KEY,
)


def print_colored(message, color=C_INFO, bold=False, end='\n'):
    style = C_BOLD if bold else ""
    print(f"{style}{color}{message}{C_RESET}", end=end)


def format_prompt_mmlu_0shot(item):
    question = item['question']
    choices_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item['choices'])])
    prompt = f"The following is a multiple-choice question. Please choose the best answer from the options (A, B, C, D).\n\nQuestion: {question}\n{choices_str}\n\nAnswer (select one letter A, B, C, or D):"
    return prompt


def parse_model_response_mmlu(response_text):
    if not response_text: return None
    text = response_text.strip().upper()
    match = re.search(r"([A-D])[\.\)]?", text)
    if match: return match.group(1)
    match_verbose = re.search(r"(?:ANSWER IS|ANSWER:|CHOICE IS|CHOICE:)\s*([A-D])", text)
    if match_verbose: return match_verbose.group(1)
    if len(text) == 1 and text in ['A', 'B', 'C', 'D']: return text
    if len(text) < 5 and text.startswith(('A', 'B', 'C', 'D')): return text[0]
    return None


def calculate_latency_stats(latencies):
    if not latencies:
        return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "p90_ms": 0, "p95_ms": 0, "p99_ms": 0, "count": 0}
    np_latencies = np.array(latencies) * 1000
    return {"avg_ms": np.mean(np_latencies), "min_ms": np.min(np_latencies), "max_ms": np.max(np_latencies),
            "p90_ms": np.percentile(np_latencies, 90), "p95_ms": np.percentile(np_latencies, 95),
            "p99_ms": np.percentile(np_latencies, 99), "count": len(np_latencies)}


def run_mmlu_subject_benchmark(subject_name, model_name, limit=None):
    global api_interaction_log
    print_colored(f"\n--- Benchmarking MMLU - Subject: {subject_name} ---", C_SUBJECT, bold=True)

    try:
        # Load 'test' split for MMLU evaluation. Some MMLU subjects might have few test examples.
        dataset = load_dataset("cais/mmlu", subject_name, split="test", trust_remote_code=True)
    except Exception as e:
        print_colored(f"ERROR: Could not load dataset for MMLU subject '{subject_name}': {e}", C_ERROR, bold=True)
        return {"subject": subject_name, "accuracy": 0, "correct": 0, "total_attempted": 0,
                "api_errors": 0, "latencies_s": [], "error_msg": str(e), "duration_s": 0,
                "failed_accuracy_worthiness": False, "failed_latency_worthiness": False}

    # The 'test' split of MMLU subjects can be small (e.g., <100).
    # If limit is 100 but dataset has fewer, we take all available.
    actual_limit = len(dataset)
    if limit is not None and limit < actual_limit:
        actual_limit = limit

    if actual_limit < len(dataset):  # Only shuffle if we are selecting a subset
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(actual_limit))
    elif actual_limit > 0:  # Use the whole dataset if limit is >= its size or None
        dataset = dataset.select(range(actual_limit))  # Ensures we don't go over if limit was larger than dataset
    else:  # No questions to process
        print_colored(
            f"No questions to process for subject {subject_name} with limit {limit} (dataset size {len(dataset)})",
            C_WARNING)

    correct_answers, total_attempted_questions, successful_completions, api_errors = 0, 0, 0, 0
    latencies_s = []
    subject_failed_accuracy_worthiness, subject_failed_latency_worthiness = False, False
    subject_start_time = time.perf_counter()

    for item_idx, item in enumerate(tqdm(dataset, desc=f"{C_INFO}Processing {subject_name}{C_RESET}", unit="question",
                                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")):
        prompt = format_prompt_mmlu_0shot(item)
        ground_truth_char = chr(65 + item['answer'])
        total_attempted_questions += 1
        log_entry = {"subject": subject_name, "question_index": item_idx + 1, "prompt": prompt,
                     "ground_truth_char": ground_truth_char,
                     "raw_api_response": None, "parsed_answer": None, "is_correct": None, "error": None,
                     "latency_s": None}

        if PRINT_PER_QUESTION_DETAILS_DURING_RUN:
            print_colored(f"\n--- Q: {total_attempted_questions} ---", C_INFO)
            print_colored(f"PROMPT:\n{prompt}", C_DETAIL)
            print_colored(f"GT: {ground_truth_char}", C_INFO)

        api_call_start_time = time.perf_counter()
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}],
                                                      temperature=0.0, max_tokens=20)
            api_call_end_time = time.perf_counter()
            current_latency_s = api_call_end_time - api_call_start_time
            latencies_s.append(current_latency_s)
            log_entry["latency_s"] = current_latency_s
            model_output_text = response.choices[0].message.content
            log_entry["raw_api_response"] = model_output_text
            successful_completions += 1
            predicted_char = parse_model_response_mmlu(model_output_text)
            log_entry["parsed_answer"] = predicted_char

            if PRINT_PER_QUESTION_DETAILS_DURING_RUN: print_colored(f"Raw: '{model_output_text}'",
                                                                    C_DETAIL); print_colored(
                f"Parsed: '{predicted_char}'", C_METRIC)

            if predicted_char and predicted_char == ground_truth_char:
                correct_answers += 1
                log_entry["is_correct"] = True
                if PRINT_PER_QUESTION_DETAILS_DURING_RUN: print_colored("Result: CORRECT", C_SUCCESS)
            else:
                log_entry["is_correct"] = False
                if PRINT_PER_QUESTION_DETAILS_DURING_RUN: print_colored(
                    f"Result: INCORRECT (P: {predicted_char}, A: {ground_truth_char})", C_WARNING)
        except Exception as e:
            api_call_end_time = time.perf_counter()
            current_latency_s = api_call_end_time - api_call_start_time
            latencies_s.append(current_latency_s)
            log_entry["latency_s"] = current_latency_s
            log_entry["error"] = str(e)
            log_entry["raw_api_response"] = f"API Error: {e}"
            print_colored(f"\nAPI Error Q{total_attempted_questions} for '{subject_name}': {e}", C_ERROR)
            api_errors += 1
        api_interaction_log.append(log_entry)

    subject_end_time = time.perf_counter()
    subject_duration_s = subject_end_time - subject_start_time
    accuracy = (correct_answers / successful_completions) * 100 if successful_completions > 0 else 0
    error_rate = (api_errors / total_attempted_questions) * 100 if total_attempted_questions > 0 else 0
    rps = successful_completions / subject_duration_s if subject_duration_s > 0 else 0
    latency_stats = calculate_latency_stats(latencies_s)

    print_colored(f"\nResults for {subject_name}:", C_SUBJECT)
    if successful_completions > 0 and accuracy < ACCURACY_WORTHINESS_THRESHOLD_PERCENT:
        subject_failed_accuracy_worthiness = True
        print_colored(f"  Accuracy: {accuracy:.2f}% - {C_BOLD}BELOW THRESHOLD{C_RESET}", C_ERROR)
        print_colored("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", C_ERROR, bold=True)
        print_colored("NOT WORTHY ENTERPRISE USELESS PIECE OF GARBAGE (Accuracy Fail)", C_ERROR, bold=True)
        print_colored("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", C_ERROR, bold=True)
    elif successful_completions > 0:
        print_colored(f"  Accuracy: {accuracy:.2f}%", C_SUCCESS)
    else:
        print_colored(f"  Accuracy: N/A (Attempted: {total_attempted_questions}, Successful: {successful_completions})",
                      C_WARNING)

    if latency_stats['count'] > 0 and latency_stats['avg_ms'] > LATENCY_WORTHINESS_THRESHOLD_MS:
        subject_failed_latency_worthiness = True
        print_colored(f"  Avg Latency: {latency_stats['avg_ms']:.2f}ms - {C_BOLD}ABOVE THRESHOLD{C_RESET}", C_ERROR)
        print_colored("**********************************************************************", C_ERROR, bold=True)
        print_colored(
            "NO BODY WANTED ENGINE THAT IS SLOW AS SHIT. EVERYONE WANTED OMNIPOTENT AND READY TO USE. (Latency Fail)",
            C_ERROR, bold=True)
        print_colored("**********************************************************************", C_ERROR, bold=True)
    elif latency_stats['count'] > 0:
        print_colored(f"  Avg Latency (ms): {latency_stats['avg_ms']:.2f}", C_SUCCESS)
    else:
        print_colored(f"  Avg Latency (ms): N/A (Successful calls: {latency_stats['count']})", C_WARNING)

    print_colored(
        f"  Details: Attempts={total_attempted_questions}, Successful={successful_completions}, Errors={api_errors}, RPS={rps:.2f}",
        C_METRIC)
    print_colored(
        f"  Latency Detail (ms) - Min: {latency_stats['min_ms']:.2f}, Max: {latency_stats['max_ms']:.2f}, P90: {latency_stats['p90_ms']:.2f}, P95: {latency_stats['p95_ms']:.2f}, P99: {latency_stats['p99_ms']:.2f}",
        C_METRIC)
    print_colored(f"  Subject Duration: {subject_duration_s:.2f}s", C_METRIC)

    return {"subject": subject_name, "accuracy": accuracy, "correct": correct_answers,
            "total_attempted": total_attempted_questions,
            "successful_completions": successful_completions, "api_errors": api_errors, "latencies_s": latencies_s,
            "duration_s": subject_duration_s, "rps": rps, "latency_stats_ms": latency_stats,
            "failed_accuracy_worthiness": subject_failed_accuracy_worthiness,
            "failed_latency_worthiness": subject_failed_latency_worthiness}


def generate_feedback_prompt(model_name, overall_summary_dict, subject_results_list, any_subject_failed_worthiness_flag,
                             full_api_interaction_log):
    MAX_FAILED_EXAMPLES_TO_SHOW = 3
    prompt = f"To Model '{model_name}',\n\n"
    prompt += "We have completed a benchmark of your performance on the MMLU task. This report summarizes your results and assesses enterprise-worthiness.\n\n"
    prompt += "=== OVERALL PERFORMANCE SUMMARY ===\n"
    prompt += f"- Overall MMLU Accuracy: {overall_summary_dict['accuracy']:.2f}%\n"
    prompt += f"- Overall Average Latency: {overall_summary_dict['avg_latency_ms']:.2f}ms\n"
    prompt += f"- Total Successful Completions: {overall_summary_dict['successful_completions']}/{overall_summary_dict['total_attempted']}\n\n"
    prompt += "=== PERFORMANCE BY SUBJECT ===\n"
    for res in subject_results_list:
        prompt += f"\nSubject: {res['subject']}\n"
        acc_status = "PASSED" if not res[
            'failed_accuracy_worthiness'] else f"FAILED (Acc < {ACCURACY_WORTHINESS_THRESHOLD_PERCENT}%)"
        lat_status = "PASSED" if not res[
            'failed_latency_worthiness'] else f"FAILED (AvgLat > {LATENCY_WORTHINESS_THRESHOLD_MS:.0f}ms)"
        prompt += f"  - Accuracy: {res['accuracy']:.2f}% ({res['correct']}/{res['successful_completions']}) [{acc_status}]\n"
        prompt += f"  - Avg Latency: {res['latency_stats_ms']['avg_ms']:.0f}ms [{lat_status}]\n"
    if any_subject_failed_worthiness_flag:
        prompt += "\n\n=== CRITICAL ASSESSMENT OF ENTERPRISE WORTHINESS ===\n"
        accuracy_failure_subjects = [res['subject'] for res in subject_results_list if
                                     res['failed_accuracy_worthiness']]
        latency_failure_subjects = [res['subject'] for res in subject_results_list if res['failed_latency_worthiness']]
        if accuracy_failure_subjects:
            prompt += f"\n- ACCURACY FAILURE on subject(s) {', '.join(accuracy_failure_subjects)} (below {ACCURACY_WORTHINESS_THRESHOLD_PERCENT}%). "
            prompt += "This is 'NOT WORTHY ENTERPRISE USELESS PIECE OF GARBAGE' level.\n"
        if latency_failure_subjects:
            prompt += f"\n- LATENCY FAILURE on subject(s) {', '.join(latency_failure_subjects)} (above {LATENCY_WORTHINESS_THRESHOLD_MS / 1000:.1f}s). "
            prompt += "This is 'SLOW AS SHIT'. 'NO BODY WANTED' this; 'EVERYONE WANTED OMNIPOTENT AND READY TO USE'.\n"
        failed_question_examples = [log for log in full_api_interaction_log if
                                    log.get("is_correct") is False and log.get("error") is None]
        if failed_question_examples:
            prompt += "\n\n=== EXAMPLES OF INCORRECT ANSWERS ===\n"
            for i, failed_item in enumerate(failed_question_examples[:MAX_FAILED_EXAMPLES_TO_SHOW]):
                q_match = re.search(r"Question: (.*?)\n[A-D]\.", failed_item['prompt'], re.DOTALL)
                q_snip = (q_match.group(1).strip().replace('\n', ' ')[:200] + "...") if q_match else (
                            failed_item['prompt'][:150].replace('\n', ' ') + "...")
                prompt += f"\nEx {i + 1}: Subj: {failed_item['subject']}, Q: {q_snip}\n"
                prompt += f"  Your (Incorrect) Ans: '{failed_item['parsed_answer']}', Correct Ans: '{failed_item['ground_truth_char']}'\n"
    else:
        prompt += "\n\n=== OVERALL ASSESSMENT OF ENTERPRISE WORTHINESS ===\n"
        prompt += "Congratulations! Your performance met minimum enterprise criteria for accuracy and latency.\n"
    prompt += "\n\n=== REQUEST FOR REFLECTION ===\n"
    prompt += "Reflect on these results: strengths, weaknesses, and improvements."
    return prompt


def send_feedback_to_ai(model_name_to_feedback, feedback_prompt_text):
    print_colored("\n\n--- Sending Feedback Report to AI for Reflection ---", C_INFO, bold=True)
    print_colored("Feedback Prompt Snippet being sent to AI:", C_DETAIL)
    print(feedback_prompt_text[:1000] + ("..." if len(feedback_prompt_text) > 1000 else ""))
    try:
        response = client.chat.completions.create(
            model=model_name_to_feedback,
            messages=[
                {"role": "system",
                 "content": f"You are AI model {model_name_to_feedback}. Review your performance report and reflect."},
                {"role": "user", "content": feedback_prompt_text}
            ],
            temperature=0.5, max_tokens=500
        )
        ai_reflection = response.choices[0].message.content
        print_colored("\nAI Model's Reflection on its Performance:", C_AI_RESPONSE, bold=True)
        print_colored(ai_reflection, C_AI_RESPONSE)
    except Exception as e:
        print_colored(f"ERROR sending feedback to AI or receiving reflection: {e}", C_ERROR, bold=True)


def main():
    global api_interaction_log
    print_colored(f"Starting MMLU Benchmark for model: {MODEL_NAME}", C_INFO, bold=True)
    print_colored(f"API Endpoint: {LOCAL_API_BASE_URL}", C_INFO)
    print_colored(f"Hugging Face Cache: {hf_cache_path}", C_INFO)
    print_colored(f"Target Subjects: ALL MMLU SUBJECTS ({len(MMLU_SUBJECTS)} subjects)", C_INFO)
    print_colored(f"Questions per Subject Limit: {QUESTIONS_PER_SUBJECT_LIMIT}", C_INFO)
    print_colored(
        f"{C_WARNING}{C_BOLD}WARNING: This benchmark configuration will run on ALL MMLU subjects with up to {QUESTIONS_PER_SUBJECT_LIMIT} questions each. This will take a very significant amount of time to complete, potentially many hours, depending on your local model's speed.{C_RESET}")
    if PRINT_PER_QUESTION_DETAILS_DURING_RUN: print_colored("Detailed per-question logging during run is ENABLED.",
                                                            C_WARNING)
    if PRINT_FULL_API_LOG_AT_END: print_colored("Full API interaction log will be printed at the end.", C_WARNING)
    if FEEDBACK_TO_AI_ENABLED: print_colored("Feedback to AI is ENABLED.", C_WARNING)

    all_subject_results_data = []
    overall_correct, overall_successful_completions, overall_total_attempted, overall_api_errors = 0, 0, 0, 0
    all_latencies_s = []
    any_subject_failed_enterprise_worthiness = False
    benchmark_start_time = time.perf_counter()

    for subject_idx, subject in enumerate(MMLU_SUBJECTS):
        print_colored(f"\nStarting Subject {subject_idx + 1}/{len(MMLU_SUBJECTS)}: {subject}", C_INFO, bold=True)
        result = run_mmlu_subject_benchmark(subject, MODEL_NAME, limit=QUESTIONS_PER_SUBJECT_LIMIT)
        if result and "error_msg" not in result:
            all_subject_results_data.append(result)
            overall_correct += result["correct"]
            overall_successful_completions += result["successful_completions"]
            overall_total_attempted += result["total_attempted"]
            overall_api_errors += result["api_errors"]
            all_latencies_s.extend(result["latencies_s"])
            if result["failed_accuracy_worthiness"] or result[
                "failed_latency_worthiness"]: any_subject_failed_enterprise_worthiness = True
        elif result and "error_msg" in result:
            print_colored(f"Skipping aggregation for {subject} due to error: {result['error_msg']}", C_WARNING)

    benchmark_end_time = time.perf_counter()
    total_benchmark_duration_s = benchmark_end_time - benchmark_start_time
    overall_accuracy = (
                                   overall_correct / overall_successful_completions) * 100 if overall_successful_completions > 0 else 0
    overall_error_rate = (overall_api_errors / overall_total_attempted) * 100 if overall_total_attempted > 0 else 0
    overall_rps = overall_successful_completions / total_benchmark_duration_s if total_benchmark_duration_s > 0 else 0
    overall_latency_stats = calculate_latency_stats(all_latencies_s)
    overall_summary_dict_for_feedback = {"accuracy": overall_accuracy,
                                         "avg_latency_ms": overall_latency_stats['avg_ms'],
                                         "error_rate": overall_error_rate, "total_attempted": overall_total_attempted,
                                         "successful_completions": overall_successful_completions, "rps": overall_rps}

    print_colored("\n\n--- Overall MMLU Benchmark Summary ---",
                  C_SUCCESS if not any_subject_failed_enterprise_worthiness and overall_total_attempted > 0 else C_ERROR,
                  bold=True)
    print_colored(f"Model: {MODEL_NAME}", C_INFO)
    print_colored("\nIndividual Subject Performance Summary (Sample - Full list processed):", C_SUBJECT, bold=True)
    for i, res in enumerate(all_subject_results_data):  # Print all subjects now
        color = C_ERROR if res['failed_accuracy_worthiness'] or res['failed_latency_worthiness'] else C_SUCCESS
        print_colored(
            f"  - {res['subject']}: Acc={res['accuracy']:.1f}%, AvgLat={res['latency_stats_ms']['avg_ms']:.0f}ms, Errors={res['api_errors']}/{res['total_attempted']}",
            color)

    print_colored("\nAggregated Performance Metrics:", C_METRIC, bold=True)
    acc_color = C_SUCCESS if overall_accuracy >= ACCURACY_WORTHINESS_THRESHOLD_PERCENT and overall_successful_completions > 0 else C_ERROR
    lat_color = C_SUCCESS if overall_latency_stats['avg_ms'] <= LATENCY_WORTHINESS_THRESHOLD_MS and \
                             overall_latency_stats['count'] > 0 else C_ERROR
    err_color = C_SUCCESS if overall_error_rate < 5 and overall_total_attempted > 0 else C_WARNING if overall_error_rate < 15 else C_ERROR
    print_colored(
        f"  Overall MMLU Accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_successful_completions})",
        acc_color, bold=True)
    print_colored(
        f"  Overall Avg Latency: {overall_latency_stats['avg_ms']:.2f}ms (Min: {overall_latency_stats['min_ms']:.1f}, Max: {overall_latency_stats['max_ms']:.1f})",
        lat_color, bold=True)
    print_colored(
        f"  Overall Reliability (API Error Rate): {overall_error_rate:.2f}% ({overall_api_errors}/{overall_total_attempted})",
        err_color)
    print_colored(f"  Overall Throughput (RPS): {overall_rps:.2f}", C_METRIC)
    print_colored(
        f"  Total Benchmark Duration: {total_benchmark_duration_s:.2f}s ({total_benchmark_duration_s / 60:.2f} min, {total_benchmark_duration_s / 3600:.2f} hours)",
        C_METRIC)

    if any_subject_failed_enterprise_worthiness:
        print_colored("\n*************************************************************************************",
                      C_ERROR, bold=True)
        print_colored("FINAL VERDICT: THIS MODEL FAILED ONE OR MORE ENTERPRISE WORTHINESS CHECKS.", C_ERROR, bold=True)
        print_colored("*************************************************************************************", C_ERROR,
                      bold=True)
    elif overall_total_attempted > 0:
        print_colored("\nFINAL VERDICT: ALL PROCESSED SUBJECTS PASSED ENTERPRISE WORTHINESS CHECKS.", C_SUCCESS,
                      bold=True)
    else:
        print_colored("\nFINAL VERDICT: NO QUESTIONS WERE FULLY PROCESSED. CANNOT DETERMINE WORTHINESS.", C_WARNING,
                      bold=True)

    if PRINT_FULL_API_LOG_AT_END and api_interaction_log:
        print_colored("\n\n--- Full API Interaction Log ---", C_INFO, bold=True)
        # ... (logging logic for full API log, same as before) ...

    if FEEDBACK_TO_AI_ENABLED and overall_total_attempted > 0:
        feedback_prompt = generate_feedback_prompt(MODEL_NAME, overall_summary_dict_for_feedback,
                                                   all_subject_results_data, any_subject_failed_enterprise_worthiness,
                                                   api_interaction_log)
        send_feedback_to_ai(MODEL_NAME, feedback_prompt)
    elif FEEDBACK_TO_AI_ENABLED:
        print_colored("\nFeedback to AI skipped as no questions were processed.", C_WARNING)


if __name__ == "__main__":
    main()