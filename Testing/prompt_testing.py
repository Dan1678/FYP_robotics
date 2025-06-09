# evaluate_planning.py

import re
import sys
import os
import traceback


from Planning.gpt_functions import extract_task_objects, generate_task_details
from Execution.client_script import generate_instructions, clean_command

# Object 1: red block
x_red_block, y_red_block, z_red_block = 1, 2, 3
roll_red_block, pitch_red_block, yaw_red_block = 4, 5, 6

# Object 2: bin
x_bin, y_bin, z_bin = 7, 8, 9
roll_bin, pitch_bin, yaw_bin = 10, 11, 12

# Object 3: green block
x_green_block, y_green_block, z_green_block = 13, 14, 15
roll_green_block, pitch_green_block, yaw_green_block = 16, 17, 18

# Object 4: apple
x_apple, y_apple, z_apple = 19, 20, 21
roll_apple, pitch_apple, yaw_apple = 22, 23, 24

# -----------------------------------------------------------------------------
# Helper functions & constants
# -----------------------------------------------------------------------------

def parse_floats_from_command(command_str):
    """
    Extract all floats (and ints) from a command string in parsing order.
    Returns a list of Python floats.
    """
    number_pattern = re.compile(r'-?\d+(\.\d+)?')
    return [float(m.group(0)) for m in number_pattern.finditer(command_str)]


def prompt_yes_no(question: str) -> bool:
    """
    Print `question`, expect user to type 'y' or 'n', return True for 'y', False for 'n'.
    Keeps asking until a valid response.
    """
    while True:
        ans = input(question + " (y/n): ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"
        print("Please type 'y' or 'n' and press Enter.")


# -----------------------------------------------------------------------------
# Test scene definition (numeric) for actual command generation
# -----------------------------------------------------------------------------

test_scene_objects_dict = {
    "red block": {
        "position": (x_red_block, y_red_block, z_red_block),
        "orientation": (roll_red_block, pitch_red_block, yaw_red_block)
    },
    "bin": {
        "position": (x_bin, y_bin, z_bin),
        "orientation": (roll_bin, pitch_bin, yaw_bin)
    },
    "green block": {
        "position": (x_green_block, y_green_block, z_green_block),
        "orientation": (roll_green_block, pitch_green_block, yaw_green_block)
    },
    "apple": {
        "position": (x_apple, y_apple, z_apple),
        "orientation": (roll_apple, pitch_apple, yaw_apple)
    },
    # Add more objects as needed…
}

# -----------------------------------------------------------------------------
# Combined prompt definitions for tight_prompt_list and free_prompt_list
# -----------------------------------------------------------------------------

# Each entry: (prompt_text, expected_object_list, expected_numeric_commands)

tight_prompt_list = [
    # --- 10 Simple “Tight” Prompts ---
    (
        "Pick up the red block and place it in the bin.",
        ["red block", "bin"],
        [
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Move the green block to the bin.",
        ["green block", "bin"],
        [
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Pick up the apple and put it in the bin.",
        ["apple", "bin"],
        [
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"pick_up({x_apple},{y_apple},{z_apple})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Retrieve the red block and drop it into the bin.",
        ["red block", "bin"],
        [
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Please move the green block from its spot to the bin.",
        ["green block", "bin"],
        [
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Lift the red block and place it inside the bin.",
        ["red block", "bin"],
        [
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Pick up the green block and place it into the bin.",
        ["green block", "bin"],
        [
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Pick up the apple and place it in the bin.",
        ["apple", "bin"],
        [
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"pick_up({x_apple},{y_apple},{z_apple})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Move the red block into the bin.",
        ["red block", "bin"],
        [
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Place the green block in the bin.",
        ["green block", "bin"],
        [
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),

    # --- 5 More Complex “Tight” Prompts ---
    (
        "Pick up the red block, then pick up the green block, and place both into the bin (red first, then green).",
        ["red block", "green block", "bin"],
        [
            # Red block → bin
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})",

            # Green block → bin
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "First, move the green block to the bin; then move the red block to the apple's location.",
        ["green block", "red block", "bin", "apple"],
        [
            # Green block → bin
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})",

            # Red block → apple position
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"place({x_apple},{y_apple},{z_apple})"
        ],
    ),
    (
        "Pick up the apple, place it in the bin, then pick up the green block and place it at the red block’s position.",
        ["apple", "green block", "red block", "bin"],
        [
            # 1. pick_up(apple)
            f"pick_up({x_apple},{y_apple},{z_apple})",

            # 2. move(apple orientation)
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",

            # 3. place → bin
            f"place({x_bin},{y_bin},{z_bin})",

            # 4. move(bin orientation)
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",

            # 5. pick_up(green block)
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",

            # 6. move(green block orientation)
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",

            # 7. place → red block’s location
            f"place({x_red_block},{y_red_block},{z_red_block})"
        ],
    ),
    (
        "Pick up the red block and apple in sequence, put the red block into the bin, then place the apple on top of the green block.",
        ["red block", "apple", "bin", "green block"],
        [
            # Red block → bin
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})",

            # Apple → green block’s position
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"pick_up({x_apple},{y_apple},{z_apple})",
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"place({x_green_block},{y_green_block},{z_green_block})"
        ],
    ),
    (
        "Move the green block next to the red block, then swap their positions without using the bin.",
        ["green block", "red block"],
        [
            # Green block → red’s position
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"place({x_red_block},{y_red_block},{z_red_block})",

            # Red block → green’s original position
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"place({x_green_block},{y_green_block},{z_green_block})"
        ],
    )
]

free_prompt_list = [
    # --- 10 Simple “Free” Prompts ---
    (
        "Hey, can you grab that red block and toss it in the bin?",
        ["red block", "bin"],
        [
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "I’d appreciate if you could move the green block over to the bin.",
        ["green block", "bin"],
        [
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Could you pick up the apple and drop it in the bin?",
        ["apple", "bin"],
        [
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"pick_up({x_apple},{y_apple},{z_apple})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Can you get that red block and put it into the bin, please?",
        ["red block", "bin"],
        [
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Do me a favor and move the green block into the bin.",
        ["green block", "bin"],
        [
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Would you pick up the apple and place it in its bin?",
        ["apple", "bin"],
        [
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"pick_up({x_apple},{y_apple},{z_apple})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Could you grab the green block and put it inside the bin?",
        ["green block", "bin"],
        [
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Please fetch the red block and set it in the bin.",
        ["red block", "bin"],
        [
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "I want that apple in the bin—can you handle that?",
        ["apple", "bin"],
        [
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"pick_up({x_apple},{y_apple},{z_apple})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),
    (
        "Can you get the green block and drop it in the bin for me?",
        ["green block", "bin"],
        [
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})"
        ],
    ),

    # --- 5 More Complex “Free” Prompts ---
    (
        "Hey, grab the red block and the green block one after the other; drop the red block in the bin first, then put the green block exactly where the red block was.",
        ["red block", "green block", "bin"],
        [
            # Red block → bin
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})",

            # Green block → red’s position
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"place({x_red_block},{y_red_block},{z_red_block})"
        ],
    ),
    (
        "Can you first move the apple to the bin, then pick up the green block and set it on the red block’s spot?",
        ["apple", "green block", "red block", "bin"],
        [
            # Apple → bin
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"pick_up({x_apple},{y_apple},{z_apple})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})",

            # Green block → red’s position
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"place({x_red_block},{y_red_block},{z_red_block})"
        ],
    ),
    (
        "I’d like you to get that red block out of the way into the bin, then take the apple and put it on top of the green block.",
        ["red block", "apple", "green block", "bin"],
        [
            # Red block → bin
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})",

            # Apple → green’s position
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"pick_up({x_apple},{y_apple},{z_apple})",
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"place({x_green_block},{y_green_block},{z_green_block})"
        ],
    ),
    (
        "Could you take both the red block and the apple, place the red block in the bin first, and then put the apple where the green block is now?",
        ["red block", "apple", "green block", "bin"],
        [
            # Red block → bin
            f"move({x_red_block},{y_red_block},{z_red_block},{roll_red_block},{pitch_red_block},{yaw_red_block})",
            f"pick_up({x_red_block},{y_red_block},{z_red_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})",

            # Apple → green’s position
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"pick_up({x_apple},{y_apple},{z_apple})",
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"place({x_green_block},{y_green_block},{z_green_block})"
        ],
    ),
    (
        "Please put the green block into the bin, then move the bin next to the apple so that it touches.",
        ["green block", "bin", "apple"],
        [
            # Green block → bin
            f"move({x_green_block},{y_green_block},{z_green_block},{roll_green_block},{pitch_green_block},{yaw_green_block})",
            f"pick_up({x_green_block},{y_green_block},{z_green_block})",
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"place({x_bin},{y_bin},{z_bin})",

            # Bin → apple location
            f"move({x_bin},{y_bin},{z_bin},{roll_bin},{pitch_bin},{yaw_bin})",
            f"pick_up({x_bin},{y_bin},{z_bin})",
            f"move({x_apple},{y_apple},{z_apple},{roll_apple},{pitch_apple},{yaw_apple})",
            f"place({x_apple},{y_apple},{z_apple})"
        ],
    )
]

# -----------------------------------------------------------------------------
# Main interactive evaluation loop
# -----------------------------------------------------------------------------

def evaluate_list(prompt_list, regime_name, model_name):
    """
    For each tuple in prompt_list:
      (prompt_text, expected_objects, expected_numeric_commands),
    interactively ask the user to verify:
      1) object extraction
      2) (if extraction is correct) command generation & comparison numeric vs expected
    Returns: (num_prompts, extraction_correct_count, move_correct_count)
    """
    total = len(prompt_list)
    extraction_correct = 0
    move_correct = 0

    print(f"\n=== Evaluating regime: {regime_name} | model: {model_name} ({total} prompts) ===\n")

    for idx, (prompt_text, expected_objects, expected_numeric_commands) in enumerate(prompt_list, start=1):
        print(f"\n[{regime_name} #{idx}/{total}] Prompt:\n  \"{prompt_text}\"")
        print("Expected object list:", expected_objects)

        # --- 1) Object extraction ---
        try:
            extracted = extract_task_objects(prompt_text)
        except Exception:
            print("Error during extract_task_objects():")
            traceback.print_exc()
            extracted = []

        print("Extracted by GPT:", extracted)
        ok_obj = prompt_yes_no("Were the extracted objects correct?")
        if ok_obj:
            extraction_correct += 1
        else:
            print("→ Marking this prompt as: object-extraction INCORRECT.\n")
            # Skip command stage if extraction was incorrect
            continue

        # --- 2) Command generation & cleaning ---
        try:
            # generate_task_details signature: generate_task_details(task, objects)
            task_details = generate_task_details(prompt_text, test_scene_objects_dict)
        except Exception:
            print("Error during generate_task_details():")
            traceback.print_exc()
            print("→ Skipping move-generation for this prompt.\n")
            continue

        try:
            # Now generate instructions with the chosen model_name
            raw_cmds = generate_instructions(task_details, model_name)
        except Exception:
            print("Error during generate_instructions():")
            traceback.print_exc()
            print("→ Skipping move-generation for this prompt.\n")
            continue

        # Attempt to clean each raw line (numeric output)
        cleaned_cmds = []
        cleaning_error = False
        for raw in raw_cmds:
            try:
                c = clean_command(raw)
                cleaned_cmds.append(c)
            except Exception:
                print("clean_command() failed on:")
                print(f"    {raw}")
                traceback.print_exc()
                cleaning_error = True
                break

        print("\nExpected numeric commands (exactly):")
        for line in expected_numeric_commands:
            print("   ", line)

        
        if not cleaning_error:
            print("\nCleaned numeric commands that WOULD be sent to the robot:")
            for line in cleaned_cmds:
                print("   ", line)
        else:
            print("\nUnable to show cleaned commands (cleaning_error).")
        
        print("\nRaw lines from GPT (numeric):")
        for line in raw_cmds:
            print("   ", line)


        if cleaning_error:
            print("→ Move-generation automatically marked INCORRECT due to cleaning error.\n")
            continue

        # Ask user if cleaned numeric output matches expected exactly
        ok_move = prompt_yes_no("Were all cleaned commands correct (exact match to expected numeric list)?")
        if ok_move:
            move_correct += 1
            print("→ Marking this prompt as: move-generation CORRECT.\n")
        else:
            print("→ Marking this prompt as: move-generation INCORRECT.\n")

    return total, extraction_correct, move_correct


def main():
    print("=== Interactive Planning Evaluation ===")

    # List of models to evaluate
    model_list = ["gpt-3.5-turbo", "gpt-4"]

    # Store results for summary
    overall_results = {}

    for model_name in model_list:
        print(f"\n**** Now evaluating with model: {model_name} ****\n")

        # Evaluate “tight” prompts with the current model
        total_tight, ext_tight, move_tight = evaluate_list(tight_prompt_list, "tight", model_name)

        # Evaluate “free” prompts with the current model
        total_free, ext_free, move_free = evaluate_list(free_prompt_list, "free", model_name)

        overall_results[model_name] = {
            "tight": (total_tight, ext_tight, move_tight),
            "free":  (total_free, ext_free, move_free)
        }

    # Print combined summary
    print("\n=== COMBINED SUMMARY ===")
    for model_name, data in overall_results.items():
        total_tight, ext_tight, move_tight = data["tight"]
        total_free, ext_free, move_free = data["free"]
        total_prompts = total_tight + total_free
        total_ext = ext_tight + ext_free
        total_move = move_tight + move_free

        print(f"\n[Model: {model_name}]")
        print(f"  Tight prompts: {ext_tight}/{total_tight} objects correct, {move_tight}/{total_tight} moves correct")
        print(f"  Free prompts:  {ext_free}/{total_free} objects correct, {move_free}/{total_free} moves correct")
        print(f"  Overall object-extraction: {total_ext}/{total_prompts} = {total_ext/total_prompts:.2%}")
        print(f"  Overall move-generation:   {total_move}/{total_prompts} = {total_move/total_prompts:.2%}")

    print("\nThank you for evaluating. Goodbye.")


if __name__ == "__main__":
    main()
