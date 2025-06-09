import openai
import json


# All ai agents and related functions are located in this file

# Initialize your OpenAI API key here
openai.api_key = ''

def extract_task_features(task_description):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Extract the key features from this task description: {task_description}"}
            ]
        )
        task_features = response.choices[0].message['content']
        return task_features
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Function to generate robot instructions from GPT
def generate_instructions(task_details, model = "gpt-4"):
    openai.api_key = ''  # Add your OpenAI API key

    prompt = f"""
    Given the following task, generate a list of robot commands according to the formats below. Each command must strictly follow the given format without any extra numbering, punctuation, or commentary. Also remember that to pick something up you must move to it first.

Task Details:
{task_details}

Requirements:
1. For a move command, output exactly six numerical values (x, y, z, roll, pitch, yaw) in the format:
   move(x, y, z, roll, pitch, yaw)
2. Whenever you move to a specific object’s position, INCLUDING THE BIN, you must fill in that object’s roll, pitch, and yaw—never re‐use the roll/pitch/yaw from a previous move.
3. For a pick_up command, output exactly three numerical values (x, y, z) in the format:
   pick_up(x, y, z)
4. For a place command, output exactly three numerical values (x, y, z) in the format:
   place(x, y, z)

Return only the commands, one per line, with no extra text or numbering. Ensure that every move command has all six coordinates and that no command is truncated.

    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )

    # Split the response into individual commands
    return response['choices'][0]['message']['content'].strip().split('\n')
    

def generate_task_details(task, objects):
    """
    Generates a formatted task details string.

    Parameters:
      - task (str): A high-level description of the task.
      - objects (dict): A dictionary where each key is an object name (e.g., "Red block", "Bin")
                        and each value is another dictionary with:
                           - "position": a tuple of three floats (x, y, z)
                           - "orientation": a tuple of three floats (roll, pitch, yaw)

    Returns:
      - details (str): A formatted multi-line string containing the task details.
    
    Example:
      task = "move to and pick up the red block. Place the block into the bin"
      objects = {
          "Red block": {"position": (81.30, -310.60, 100.00), "orientation": (74.31, 0.13, -5.22)},
          "Bin": {"position": (172.8, -226.4, 107.4), "orientation": (93.9, -0.83, 47.41)}
      }
      
      The function will return a string in the following format:
      
      Task: move to and pick up the red block. Place the block into the bin
      
      Red block:
      Position: 81.3  -310.6  100.0
      Orientation: 74.31  0.13  -5.22  # Roll, Pitch, Yaw
      
      Bin:
      Position: 172.8  -226.4  107.4
      Orientation: 93.9  -0.83  47.41  # Roll, Pitch, Yaw
    """
    details = f"Task: {task}\n\n"
    for obj_name, data in objects.items():
        pos = data.get("position", ("N/A", "N/A", "N/A"))
        orient = data.get("orientation", ("N/A", "N/A", "N/A"))
        details += f"{obj_name}:\n"
        details += f"Position: {pos[0]}  {pos[1]}  {pos[2]}\n"
        details += f"Orientation: {orient[0]}  {orient[1]}  {orient[2]}  # Roll, Pitch, Yaw\n\n"
    return details

def extract_task_objects(task_description):
    """
    Uses GPT to extract a list of task objects from the task description.
    Returns a JSON array of strings.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": (
                    f"Extract the list of objects mentioned in the following task description. "
                    "Return a JSON array of strings with each object name, and do not include any numbering or extra text.\n\n"
                    f"Task Description: {task_description}"
                )}
            ]
        )
        content = response.choices[0].message['content']
        # Parse the JSON response into a list
        objects = json.loads(content)
        return objects
    except Exception as e:
        print(f"An error occurred in extract_task_objects: {e}")
        return []

def main():
    task_description = input("Please enter your task description: ")
    features = extract_task_features(task_description)
    print(f"Extracted features: {features}")

def generate_open_verification_prompt(
    task_description: str,
    table_confidences: dict,
    table_poses: dict,
    bin_confidences: dict,
    bin_poses: dict,
    bin_fixed_poses: dict
) -> str:
    """
    Build an open‐ended verification prompt. We show:
      • Original task.
      • Table‐view confidence & pose for each object.
      • Bin‐view confidence & pose for each object.
      • Known bin coordinates.
      • For each object, an explicit “table_confidence = X, bin_confidence = Y, and because Y > X, that object is in the bin” example.
      • A rule to map any reported bin‐view pose (within ±0.2) to its bin.
      • Finally, ask GPT to decide success/failure and provide retry instructions if needed.
    """

    lines = []

    # 1) Original task
    lines.append(f"Original task:\n“{task_description}”\n")

    # 2) Table‐view observations
    lines.append("Table‐view observations:")
    for obj, conf in table_confidences.items():
        if table_poses.get(obj) is not None:
            x, y, z, r, p, yaw = table_poses[obj]
            lines.append(
                f"  • {obj}: table_confidence = {conf:.2f}; pose = "
                f"(X={x:.2f}, Y={y:.2f}, Z={z:.2f}; "
                f"roll={r:.2f}, pitch={p:.2f}, yaw={yaw:.2f})"
            )
        else:
            lines.append(f"  • {obj}: table_confidence = {conf:.2f} (not detected on table)")
    lines.append("")

    # 3) Bin‐view observations
    lines.append("Bin‐view observations:")
    for obj, conf in bin_confidences.items():
        if bin_poses.get(obj) is not None:
            x, y, z, r, p, yaw = bin_poses[obj]
            lines.append(
                f"  • {obj}: bin_confidence = {conf:.2f}; pose = "
                f"(X={x:.2f}, Y={y:.2f}, Z={z:.2f}; "
                f"roll={r:.2f}, pitch={p:.2f}, yaw={yaw:.2f})"
            )
        else:
            lines.append(f"  • {obj}: bin_confidence = {conf:.2f} (not detected in bin)")
    lines.append("")

    # 4) Known bin coordinates (for reference)
    lines.append("Known bin coordinates (for reference):")
    for bin_name, data in bin_fixed_poses.items():
        bx, by, bz = data["position"]
        br, bp, byw = data["orientation"]
        lines.append(
            f"  • {bin_name}: "
            f"(X={bx:.2f}, Y={by:.2f}, Z={bz:.2f}; "
            f"roll={br:.2f}, pitch={bp:.2f}, yaw={byw:.2f})"
        )
    lines.append("")

    # 5) Explicit per‐object numeric comparison
    lines.append("Per‐object comparison examples (using the exact numbers above):")
    for obj in table_confidences:
        t_conf = table_confidences[obj]
        b_conf = bin_confidences.get(obj, 0.0)
        # Spell out the comparison
        lines.append(
            f"  • {obj}: table_confidence = {t_conf:.2f}, bin_confidence = {b_conf:.2f} "
            f"→ since {b_conf:.2f} {'>' if b_conf > t_conf else '<='} {t_conf:.2f}, "
            f"{obj} is {'in the bin' if b_conf > t_conf else 'on the table'}."
        )
    lines.append("")

    # 6) Matching rule for bin‐view pose → bin name
    lines.append(
        "Also note: If any object's reported bin‐view pose is within ±0.2 units of one of the "
        "\"Known bin coordinates\" above, treat that object as being in that bin. "
        "For example, any pose near (X=172.80, Y=-226.40, Z=107.40; roll=93.90, pitch=-0.83, yaw=47.41) "
        "means Green Bin."
    )
    lines.append("")

    # 7) Open‐ended instructions to GPT
    lines.append(
        "Using the information, the explicit comparisons, and the rules above, please determine "
        "whether each object has been placed correctly according to the original task. If all objects "
        "ended up in the correct bin, respond with “Yes, the task is complete.” If any object remains "
        "on the table or is in the wrong bin, respond with “No,” followed by a brief explanation naming "
        "which object(s) remain or are misplaced and clear instructions on how to retry (for example: "
        "“Place the lemon into the Green Bin.”). Do not include raw numeric poses in your response. "
        "Be concise and descriptive. Treat each object independently."
    )
    lines.append("")

    return "\n".join(lines)



def chat_with_gpt(prompt: str, model: str = "gpt-4") -> str:
    """
    Sends 'prompt' to GPT and returns the assistant’s text response.

    Parameters:
      - prompt (str): The full text prompt to send.
      - model  (str): The OpenAI model name to use (default: "gpt-3.5-turbo").

    Returns:
      - The assistant’s reply as a string.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"An error occurred in chat_with_gpt: {e}")
        return ""


def parse_verification_response(response_text: str) -> dict:
    """
    (Optional helper) Attempt to extract from GPT’s verification response:
      - A list of misplaced objects (if any).
      - For each, the 6D pick and 6D place pose recommended.

    This function returns a dict of the form:
      {
        "misplaced_objects": {
          "object_name": {
            "pick_pose": (x, y, z, roll, pitch, yaw),
            "place_pose": (x, y, z, roll, pitch, yaw)
          },
          ...
        }
      }

    If parsing fails or GPT did not recommend any retries, returns an empty dict.

    Note: GPT can return free‐form text. To reliably parse out exact 6D numbers, in practice
    you would prompt GPT to respond in strict JSON. For now, this is a placeholder that simply
    returns the raw text under “raw_response” so you can manually inspect or apply a custom parser.

    Example usage:
      result = parse_verification_response(gpt_reply)
      if "misplaced_objects" in result:
          for obj, poses in result["misplaced_objects"].items():
              pick = poses["pick_pose"]
              place = poses["place_pose"]
              # … use pick/place in your retry …
    """
    # Placeholder: do not attempt actual numeric parsing here.
    return {"raw_response": response_text}


# … rest of your existing file …


if __name__ == "__main__":
    main()
