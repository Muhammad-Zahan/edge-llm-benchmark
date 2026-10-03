# Edge LLM Benchmark

I built this during my internship at Embotics to check one simple thing: can a small language model run on a low-power device like a Raspberry Pi 5, and how fast is it?

This repo has the script I used to test Qwen2.5-1.5B in FP16 format on the CPU with OpenVINO GenAI.

## What the script does

- Loads the model and starts a chat in the terminal.
- Streams the answer word by word as it is generated.
- After every answer, prints:
  - number of tokens generated
  - time to first token (TTFT), which is how long you wait before the answer starts
  - total time
  - speed in tokens per second

## Setup

1. Install the packages:

```
pip install -r requirements.txt
```

2. Convert the model to OpenVINO format:

```
optimum-cli export openvino --model Qwen/Qwen2.5-1.5B-Instruct --weight-format fp16 qwen2.5-1.5b-fp16-ov
```

3. Run the script:

```
python "QWEN2.5 FP16.py"
```

Type your question and press Enter. Type `exit` to stop.

## Settings used

| Setting | Value |
|---|---|
| Max new tokens | 200 |
| Temperature | 0.7 |
| Top-p | 0.9 |
| Top-k | 50 |
| Repetition penalty | 1.1 |
| Device | CPU |

## Results

Qwen2.5-1.5B, FP16:

| Metric | Value |
|---|---|
| Time to first token | 834 ms |
| Speed | 7.9 tokens/sec |

Screenshots of the runs are in the `Results/` folder.

## Things to know

- Speed is total tokens divided by total time, so it includes the time to process the prompt. Pure generation speed is a bit higher.
- The script keeps the chat history, so later questions in the same session can have a slightly longer first-token time.
- The model files are not in this repo because they are about 3 GB. Use the convert command above to make them.