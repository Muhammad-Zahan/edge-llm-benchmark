import os,time
import openvino_genai as ov

path="qwen2.5-1.5b-fp16-ov"
dev="CPU"
sys_msg="You are a helpful, knowledgeable assistant. Answer clearly and concisely, and ask for clarification if a request is ambiguous."

print(f"Loading {path} on {dev}...")
pipe=ov.LLMPipeline(path,dev,INFERENCE_NUM_THREADS=os.cpu_count())
print("Model loaded. Type exit to quit.\n")

cfg=ov.GenerationConfig()
cfg.max_new_tokens=200
cfg.temperature=0.7
cfg.top_p=0.9
cfg.top_k=50
cfg.repetition_penalty=1.1
cfg.do_sample=True

n=0
t_first=None

def stream(s):
    global n,t_first
    if t_first is None:t_first=time.time()
    n+=1
    print(s,end="",flush=True)
    return False

pipe.start_chat(system_message=sys_msg)

while True:
    q=input("You: ").strip()
    if q.lower() in ("exit","quit"):break
    if not q:continue
    n=0
    t_first=None
    t0=time.time()
    print("Qwen: ",end="",flush=True)
    pipe.generate(q,cfg,stream)
    tot=time.time()-t0
    ttft=(t_first-t0) if t_first else 0
    tps=n/tot if tot>0 else 0
    print("\n---")
    print(f"Tokens     : {n}")
    print(f"TTFT       : {ttft*1000:.1f} ms")
    print(f"Total time : {tot:.2f} s")
    print(f"Speed      : {tps:.2f} tok/s")
    print("---\n")

pipe.finish_chat()