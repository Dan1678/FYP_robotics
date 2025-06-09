# GPT-Powered Multi-Robot Control

## Overview  
This repository implements a closed-loop multi-robot pick-and-place pipeline driven by large language models (LLMs) and vision networks. It combines:

- **Segmentation & Matching**  
  - **SAM2** for instance-agnostic mask generation  
  - **CLIP** for grounding masks to object names  
- **Task Planning**  
  - GPT-3.5/GPT-4 prompts to extract task objects, generate low-level motion commands, and decompose complex tasks  
- **Execution**  
  - TCP-based control scripts for one or two robots  
- **Verification & Retry**  
  - A second “verification” agent that confirms placements via vision and issues retries on failure  

## Entry points  
- Single-robot baseline (`single_robot_system.py`)
- Dual-robot (`dual_robot_system.py`)


## Dependencies

### Standard Library (built-in)
- `socket`
- `time`
- `json`
- `re`
- `pickle`

### Third-Party Packages
- `openai` == 0.28.1
- `opencv-python` (`cv2`) == 4.10.0.84
- `numpy` == 1.26.1
- `torch` == 2.5.0
- `Pillow` == 11.0.0
- `matplotlib` == 3.8.0

### External Repositories (SAM2 should be cloned into project root)
- `clip` – [OpenAI CLIP](https://github.com/openai/CLIP)
- `SAM2` – [FacebookResearch SAM2](https://github.com/facebookresearch/sam2
