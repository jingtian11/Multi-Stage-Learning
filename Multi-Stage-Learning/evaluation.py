import json
import argparse
from vllm import LLM, SamplingParams
import gc
import torch
import os
from vllm import LLM, SamplingParams
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="VLLM Batch Inference Script")
    parser.add_argument("--old-model-path", type=str, required=True, help="Path to the old model")
    parser.add_argument("--new-model-path", type=str, required=True, help="Path to the new model")
    parser.add_argument("--template", type=str, required=True, help="Path to the template file")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input file (jsonl format)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--step", type=int, required=True, help="Step number")
    parser.add_argument("--max-batch-size", type=int, default=4, help="Maximum batch size for parallel inference")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    return parser.parse_args()

def read_input_file(input_file):
    questions = []
    true_answers = []
    try:
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                questions.append(data["Question"])
                true_answers.append(data['Answer'])
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        raise
    return questions, true_answers

def calculate_metrics(results, true_answers):
    correct_old = 0
    correct_new = 0
    incorrect_old = 0
    correct_new_on_incorrect_old = 0
    
    for result, true_answer in zip(results, true_answers):
        old_model_answer = result['old_model_answer'].strip().lower()
        new_model_answer = result['new_model_answer'].strip().lower()
        true_answer = true_answer.strip().lower()
        
        if true_answer in old_model_answer:
            correct_old += 1
        if true_answer in new_model_answer:
            correct_new += 1
        if true_answer not in old_model_answer:
            incorrect_old += 1
            if true_answer in new_model_answer:
                correct_new_on_incorrect_old += 1
    
    old_acc = correct_old / len(true_answers)
    new_acc = correct_new / len(true_answers)
    kgr = correct_new_on_incorrect_old / incorrect_old if incorrect_old > 0 else 0  # Handle division by zero
    print("--------------------------------")
    print("print metrics")
    print(f"old_acc: {old_acc}, new_acc: {new_acc}, kgr: {kgr}")
    print("--------------------------------")
    return old_acc, new_acc, kgr

def write_output_file(output_file, results):
    logging.info(f"Writing output file: {output_file}")
    try:
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        logging.info(f"Successfully wrote to {output_file}")
    except Exception as e:
        logging.error(f"Error writing output file: {e}")
        raise

def write_metrics_file(metrics_file, old_acc, new_acc, kgr, args):
    logging.info(f"Writing metrics file: {metrics_file}")
    try:
        with open(metrics_file, 'a') as f:
            f.write(f"Step: {args.step}, Old Model Path: {args.old_model_path}, New Model Path: {args.new_model_path}, Max Batch Size: {args.max_batch_size}\n")
            f.write(f"Old ACC: {old_acc}, New ACC: {new_acc}, KGR: {kgr}\n\n")
        logging.info(f"Successfully wrote to {metrics_file}")
    except Exception as e:
        logging.error(f"Error writing metrics file: {e}")
        raise

def run_inference(model_path, prompts, sampling_params, max_batch_size):
    logging.info(f"Running inference with model: {model_path}")
    try:
        llm = LLM(model=model_path, gpu_memory_utilization=0.8, tensor_parallel_size=1, enforce_eager=True)
        results = []
        # Process prompts in batches
        for i in range(0, len(prompts), max_batch_size):
            batch_prompts = prompts[i:i + max_batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                results.append({
                    "prompt": prompt,
                    "generated_text": generated_text
                })
        return results
    except Exception as e:
        logging.error(f"Error during inference with model {model_path}: {e}")
        raise

def main():
    args = parse_args()
    
    try:
        import time
        start_time = time.time()
        
        # 读取输入文件
        questions, true_answers = read_input_file(args.input_file)
        if args.template == "llama3":
            prompts = [f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for q in questions]
        elif args.template == "mistral":
            prompts = [f"<s> [INST] {q} [/INST] " for q in questions]
        else:
            raise ValueError(f"Invalid template: {args.template}")  
        # 采样参数
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=256)
        
        # 旧模型推理
        logging.info("Inference with old model")
        old_results = run_inference(args.old_model_path, prompts, sampling_params, args.max_batch_size)
        
        # 新模型推理
        logging.info("Inference with new model")
        new_results = run_inference(args.new_model_path, prompts, sampling_params, args.max_batch_size)
        
        # 合并结果
        combined_results = []
        for i in range(len(questions)):
            combined_results.append({
                "original_question": questions[i],
                "prompt": prompts[i],
                "old_model_answer": old_results[i]['generated_text'],
                "new_model_answer": new_results[i]['generated_text']
            })
        
        # 写入结果文件
        output_file = os.path.join(args.output_dir, f"Step{args.step}_answer.jsonl")
        write_output_file(output_file, combined_results)
        
        # 计算指标
        old_acc, new_acc, kgr = calculate_metrics(combined_results, true_answers)
        
        # 写入指标文件
        metrics_file = os.path.join(args.output_dir, "metrics.txt")
        write_metrics_file(metrics_file, old_acc, new_acc, kgr, args)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Execution time: {elapsed_time:.6f} seconds")
    
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()