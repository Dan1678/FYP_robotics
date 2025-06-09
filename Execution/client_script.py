import socket
import re
from Planning.gpt_functions import generate_instructions



def clean_command(command):
    # Remove leading numbering (e.g. "1. " or "1) ")
    command = re.sub(r'^\d+[\.\)]\s*', '', command.strip())
    command = re.sub(r'\s+', ' ', command)            # Replace multiple spaces with a single space
    command = re.sub(r'\s*\(\s*', '(', command)         # Remove spaces before '('
    command = re.sub(r'\s*\)\s*', ')', command)         # Remove spaces after ')'
    command = re.sub(r'\s*,\s*', ',', command)           # Remove spaces around commas
    if command.startswith("move") or command.startswith("pick_up") or command.startswith("place"):
        return command
    else:
        raise ValueError(f"Invalid command: {command}")


def send_command_to_robot(client_socket, command):
    try:
        cleaned_command = clean_command(command)
        print(f"Sending cleaned command: {cleaned_command}")
        # Send the command terminated by a newline
        client_socket.sendall(cleaned_command.encode('utf-8') + b'\n')
        # Wait for acknowledgement from the server
        ack = client_socket.recv(1024).decode('utf-8').strip()
        print(f"Received ack: {ack}")
    except Exception as e:
        print(f"Error sending command: {e}")

def main():

    # A short script used for testing connections
    
    task_details = """
    Task: move to and pick up the red block. Place the block into the bin

    Red block:
    Position: (81.30  -310.60  100.00)
    Orientation: (74.31  0.13  -5.22)  # Roll, Pitch, Yaw

    Bin:
    Position: (172.8  -226.4 107.4)
    Orientation: (93.9 -0.83 47.41)  # Roll, Pitch, Yaw

    """

    # Generate instructions from GPT based on the task details
    instructions = generate_instructions(task_details)
    print("Generated instructions:", instructions)

    # Connect to the server and keep the connection open
    server_ip = 'xxxx'  # Replace with your Raspberry Pi's IP address
    server_port = 'xxxx'           # The port the server is listening on

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_ip, server_port))
        # Send each instruction sequentially, waiting for ack each time
        for instruction in instructions:
            send_command_to_robot(client_socket, instruction)

if __name__ == "__main__":
    main()
