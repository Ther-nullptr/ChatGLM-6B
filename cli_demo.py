import os
import platform
import argparse
from transformers import AutoTokenizer, AutoModel
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == '__main__':
    # command line demo
    parser = argparse.ArgumentParser()
    parser.add_argument("--presicion", type=str, default="int4", help="presicion of model", choices=["int4", "int8", "fp16"])
    parser.add_argument("--profile", action="store_true", help="profile the model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    if args.presicion == "int4":
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(4).cuda()
    elif args.presicion == "int8":
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(8).cuda()
    else:
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    print(model)

    os_name = platform.system()

    history = []
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        
        if args.profile:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                with record_function("model_inference"):
                    response, history = model.chat(tokenizer, query, history=history)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            # write to file
            prof.export_chrome_trace("trace.json")
        else:
            response, history = model.chat(tokenizer, query, history=history)
        print(f"ChatGLM-6B：{response}")
