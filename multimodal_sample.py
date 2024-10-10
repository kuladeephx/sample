import requests
import pandas as pd
import gc
import torch
from memory_profiler import profile
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from huggingface_hub import login, logout

hf_key = ''





def break_list_into_chunks(full_list, chunk_size):
    chunked_list = []

    # Iterate over the input list and add chunks to the result list
    for i in range(0, len(full_list), chunk_size):
        chunked_list.append(full_list[i:i + chunk_size])

    return chunked_list



@profile
def run_multimodal_model(prompt_list, llm_batch_size):
    login(token=hf_key)
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 4},
        
    )
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
    final_out_list = []
    # Breaking prompts into chunks
    prompt_chunks = break_list_into_chunks(prompt_list, llm_batch_size)
    
    del prompt_list
    for prompt_chunk in prompt_chunks:
        
        outputs = llm.generate(prompt_chunk, sampling_params=sampling_params)
        torch.cuda.empty_cache()
        gc.collect()
        

        outputs = [i.outputs[0].text.strip() for i in outputs]
        final_out_list.extend(outputs)
        del outputs
        del prompt_chunk

    
    
    llm = None
    
    del llm

    # Clear the GPU cache
    torch.cuda.empty_cache()

    return final_out_list



def generate_prompt(image_url):
    try:
        # Reading the image from url
        image_val = Image.open(requests.get(image_url, stream=True).raw)
        image1 = image_val
        image2 = image_val
    except:
        return -1
    
    prompt_val = f"""
<|user|>
<|image_1|>
<|image_2|>
Are these two image same? Just answer yes or no.
"""

    final_prompt = {
        "prompt": prompt_val,
        "multi_modal_data": {"image": [image1, image2]},
    }

    return final_prompt


@profile
def get_final_output(df, total_no_of_prompts, llm_batch_size):
    unique_image_urls = df["url"].unique().tolist()
    #Subsetting number of prompts
    unique_image_urls = sorted(unique_image_urls)[:total_no_of_prompts]
    
    print("---------------Generating Pompts---------------")
    prompts_list = [generate_prompt(image_url) for image_url in tqdm(unique_image_urls)]
    prompts_list = [x for x in prompts_list if x!=-1]
    
    final_outputs = run_multimodal_model(prompts_list,llm_batch_size)
    return final_outputs


if __name__ == "__main__":

    input_df = pd.read_csv("dummy_image_data.zip")
    
    
    result = get_final_output(df =input_df,
                              total_no_of_prompts = 300,
                              llm_batch_size = 150)
    
    """
    df: input dataframe
    total_no_of_prompts: total number of prompts to run
    llm_batch_size: no of prompts to be executed by LLM at once
    """