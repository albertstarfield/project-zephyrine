# semicolonGeminiPatcher.py
import re
import io
import sys
import os
import argparse # For command-line arguments
from loguru import logger # Using logger for script output

# Configure logger for the script itself
logger.remove()
logger.add(sys.stderr, level="INFO")

def fix_semicolon_chaining(python_code: str) -> str:
    """
    Attempts to fix Python code where semicolons are incorrectly used
    to chain statements on a single line. Tries to preserve indentation
    and ignore semicolons within various string literal types.

    Args:
        python_code: The input Python code string with potential errors.

    Returns:
        The corrected Python code string.
    """
    corrected_lines = []
    current_indent = ""

    lines = python_code.splitlines()
    logger.info(f"Processing {len(lines)} lines...")

    for i, line in enumerate(lines):
        line_num = i + 1
        stripped_line = line.lstrip()
        indent = line[:len(line) - len(stripped_line)]

        # --- Skip empty lines or pure comment lines ---
        if not stripped_line or stripped_line.startswith('#'):
            corrected_lines.append(line)
            continue

        # --- Simple check: If no semicolon outside of likely strings, keep line as is ---
        # This is a quick heuristic to avoid complex regex on simple lines
        if ';' not in line or (line.count(';') == line.count('";"') + line.count("';'")):
             # Rudimentary check assumes semicolons only appear inside basic quoted strings
             corrected_lines.append(line)
             continue

        logger.debug(f"Line {line_num}: Potentially needs splitting: {line.strip()}")

        # --- Use Regex to split by semicolons NOT inside quotes ---
        # Regex tries to capture string literals first, then semicolons
        # NOTE: This regex is complex and may have edge cases with nested/escaped quotes.
        parts = re.split(r"""
            (              # Start capturing group 1 (string literal)
              # Triple quoted strings (non-greedy content)
              (?:'''(?:\\.|'(?!'')|[^'])*?''') |
              (?:\"\"\"(?:\\.|"(?!"")|[^"])*?\"\"\") |
              # Single quoted strings (handling escaped quotes/backslashes)
              (?:'(?:\\.|[^'\\])*') |
              # Double quoted strings (handling escaped quotes/backslashes)
              (?:"(?:\\.|[^"\\])*")
            )              # End capturing group 1
            |              # OR
            (;+)           # Start capturing group 2 (one or more semicolons)
            """, line, flags=re.VERBOSE) # Split the original line including indentation

        output_line_parts = [] # Store parts for the current logical line before splitting
        current_line_indent = indent # Start with original line indent

        for part_idx, part in enumerate(parts):
            if part is None:
                continue

            # Check if the part is purely semicolons (captured by group 2)
            is_separator = part and all(c == ';' for c in part)

            if is_separator:
                # Found a separator outside a string, process the collected parts
                if output_line_parts:
                    corrected_lines.append("".join(output_line_parts))
                    logger.debug(f"  Line {line_num}: Split statement: {''.join(output_line_parts).strip()}")
                    output_line_parts = [current_line_indent] # Start next line with same indent
                # else: Handle case like `;;`? Ignore for now.
            else:
                # It's a code segment or a string literal
                if not output_line_parts:
                    # First part of a potentially new line, preserve original indent
                    output_line_parts.append(part) # Keep original spacing/indent
                    current_line_indent = line[:len(line) - len(line.lstrip())] # Capture indent
                else:
                    # Subsequent part after a semicolon, needs indent + part
                    output_line_parts.append(part)

        # Add any remaining parts for the last statement on the original line
        if output_line_parts:
            corrected_lines.append("".join(output_line_parts))
            # Log if the last segment wasn't the full original line (meaning a split happened)
            if len(parts) > 1 and not all(p is None or all(c==';' for c in p) for p in parts[1:]):
                 logger.debug(f"  Line {line_num}: Added last segment: {''.join(output_line_parts).strip()}")


    logger.info("Finished processing.")
    return "\n".join(corrected_lines)

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Corrects Python files where semicolons are used for statement chaining, preserving indentation and ignoring semicolons in strings."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input Python file (e.g., app.py)."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output file. If omitted, prints to standard output."
    )
    parser.add_argument(
        "-i", "--in-place",
        action="store_true",
        help="Modify the input file directly (USE WITH CAUTION - BACKUP FIRST!)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging."
    )


    args = parser.parse_args()

    # Set log level based on verbose flag
    if args.verbose:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
        logger.info("Verbose logging enabled.")
    else:
         logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])


    input_filepath = args.input_file
    output_filepath = args.output
    in_place = args.in_place

    if not os.path.isfile(input_filepath):
        logger.error(f"Input file not found: {input_filepath}")
        sys.exit(1)

    if in_place and output_filepath:
        logger.error("Cannot use --in-place (-i) and --output (-o) together.")
        sys.exit(1)

    logger.info(f"Processing file: {input_filepath}...")
    try:
        with open(input_filepath, "r", encoding='utf-8') as f:
            original_code = f.read()

        corrected_code = fix_semicolon_chaining(original_code)

        if in_place:
            if original_code == corrected_code:
                logger.info("No semicolon chaining issues found. File not modified.")
            else:
                 logger.warning(f"Modifying file in-place: {input_filepath}")
                 try:
                     # Create a backup before overwriting
                     backup_path = input_filepath + ".bak"
                     logger.info(f"Creating backup: {backup_path}")
                     shutil.copy2(input_filepath, backup_path)

                     with open(input_filepath, "w", encoding='utf-8') as f:
                         f.write(corrected_code)
                     logger.success("File modified successfully.")
                 except Exception as e:
                      logger.error(f"Error writing back to input file: {e}")
                      sys.exit(1)
        elif output_filepath:
            logger.info(f"Writing corrected code to: {output_filepath}")
            try:
                with open(output_filepath, "w", encoding='utf-8') as f:
                    f.write(corrected_code)
                logger.success("Corrected code written successfully.")
            except Exception as e:
                 logger.error(f"Error writing to output file: {e}")
                 sys.exit(1)
        else:
            # Print to standard output if no output file specified
            print("\n--- Corrected Code ---")
            print(corrected_code)
            print("----------------------")


    except FileNotFoundError:
        logger.error(f"Input file not found during processing: {input_filepath}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.exception("Traceback:")
        sys.exit(1)