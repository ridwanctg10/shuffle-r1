
import time
import re
import os

from utils.eval_templates import (
    get_gpt4_ICE, 
    get_gpt4_score_ICE, 
    get_gpt4_chartqa_score_ICE, 
    get_gpt4_logicvista_score_ICE,
    get_gpt4_r1_onevision_score_ICE,
    get_gpt4_extract_ICE
)

# import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part



def build_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
""" # noqa
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt

def build_chartqa_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a chart understanding question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
""" # noqa
    demo_prompt = task_description
    # examples = get_gpt4_score_ICE()
    examples = get_gpt4_chartqa_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt

def build_logicvista_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a logical reasoning question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
""" # noqa
    demo_prompt = task_description
    # examples = get_gpt4_score_ICE()
    examples = get_gpt4_logicvista_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt

def build_r1_onevision_score_prompt(question, extract, answer):
    task_description = """
Below are two answers to a STEM related reasoning question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
""" # noqa
    demo_prompt = task_description
    # examples = get_gpt4_score_ICE()
    examples = get_gpt4_r1_onevision_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    Please output the judgement score directly with no explanation.
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer]: {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt


def build_extract_prompt(prediction, question):
    task_description = """
Please read the following example.
Then output the answer extracted from the model response directly. No "Extracted answer:" in your answer.\n
"""
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def build_wemath_extract_prompt(extraction: str, question: str) -> str:
    prompt = f"""You are evaluating answers to math questions. Extract the final answer from the text.

Question: {question}

Model's solution:
{extraction}

Extract the final answer as a single letter (A, B, C, D, or E) without any explanation or other text.
If the final answer is not clear, make your best determination based on the reasoning provided.
Your output should be ONLY the letter corresponding to the answer choice.
"""
    return prompt

def build_mathverse_extract_prompt(prediction):
    task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt

def build_chartqa_extract_prompt(prediction):
    task_description = """
I am providing you a response from a model to a chart understanding problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt

def build_logicvista_extract_prompt(extraction: str, question: str) -> str:
    prompt = f"""You are a information extractor that extracts multiple choice letter answer choices from a paragraph that contains the answer choice and sometimes and explaination of why that choice is correct to the following question:

Question: {question}

Model's solution:
{extraction}

What letter did the following answer choose? If the answer did not select a letter answer choice, first try to infer the answer based off the given choices.
If it does not seem like the given answer corresponds to an answer choice OR if there is no selected answer, please just respond with the string ZZZZZ.
For example, if the model's solution is A, then you should output 'A'
If the model's solution is B,D, then you should output 'B, D'
Make sure you answer with ONLY the letters chosen.
"""
    return prompt

def build_r1_onevision_extract_prompt(prediction):
    task_description = """
I am providing you a response from a model to a STEM reasoning problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt

def extract_boxed_answer(text):
    """Extract the last boxed answer from generated text, if present."""
    boxed_matches = re.findall(r'\\boxed{([^}]+)}', text)
    if boxed_matches:
        return boxed_matches[-1].strip(), True # Return the last match
    return text, False

def retry_with_backoff(func, max_retries=3, initial_delay=2, *args, **kwargs):
    """
    通用的自动重试机制，带指数退避。
    
    参数:
        func: 需要调用的函数
        max_retries: 最大重试次数
        initial_delay: 初始重试间隔时间（秒）
        *args: 传递给函数的参数
        **kwargs: 传递给函数的关键字参数
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"尝试第 {attempt + 1} 次失败: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # 指数退避
            else:
                print("已达到最大重试次数，操作失败。")
                raise

def llm_eval_score_retry(question, prediction, answer, dataset):
    
    # NOTE: set your Vertexai environment ready before starting evaluation
    # Initialize Vertex AI and Gemini
    # You can change the following lines to enable your own API service
    PROJECT_ID = "YOUR_PROJECT_ID"
    vertexai.init(project=PROJECT_ID, location="us-central1")
    model = GenerativeModel("gemini-2.0-flash-001")
    # End of Vertex AI and Gemini initialization

    if dataset.lower() == "mathverse":
        extracted_answer, boxed_flag = extract_boxed_answer(prediction)
        if not boxed_flag:
            extract_prompt = build_mathverse_extract_prompt(prediction)
            
            # 使用重试机制调用模型
            extracted_answer = retry_with_backoff(
                model.generate_content,
                max_retries=3,
                initial_delay=2,
                contents=extract_prompt,
                generation_config={"temperature": 0.0}
            ).text

        score_prompt = build_score_prompt(question, extracted_answer, answer)
        
        # 直接调用生成内容，依靠外部重试逻辑
        response_text = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=score_prompt,
            generation_config={"temperature": 0.0}
        ).text.strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0

    elif dataset.lower() in ["mathvista", "mathvision"]:
        extract_prompt = build_extract_prompt(prediction, question)
        
        # 使用重试机制调用模型
        extracted_answer = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=extract_prompt,
            generation_config={"temperature": 0.0}
        ).text

        if extracted_answer.strip() == answer:
            return 1.0
        else:
            return 0.0

    elif dataset.lower() == "wemath":
        extract_prompt = build_wemath_extract_prompt(prediction, question)
        
        # 使用重试机制调用模型
        response = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=extract_prompt,
            generation_config={"temperature": 0.0}
        )
        
        extracted_answer = response.text.strip().upper()
        
        if re.match(r'^[A-G]$', extracted_answer):
            accuracy = 1.0 if extracted_answer == answer else 0.0
            return accuracy
        else:
            return 0.0

    elif dataset.lower() == "chartqa":
        extracted_answer, boxed_flag = extract_boxed_answer(prediction)
        if not boxed_flag:
            # extract_prompt = build_mathverse_extract_prompt(prediction)
            extract_prompt = build_chartqa_extract_prompt(prediction)
            
            # 使用重试机制调用模型
            extracted_answer = retry_with_backoff(
                model.generate_content,
                max_retries=3,
                initial_delay=2,
                contents=extract_prompt,
                generation_config={"temperature": 0.0}
            ).text

        # score_prompt = build_score_prompt(question, extracted_answer, answer)
        score_prompt = build_chartqa_score_prompt(question, extracted_answer, answer)
        
        # 直接调用生成内容，依靠外部重试逻辑
        response_text = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=score_prompt,
            generation_config={"temperature": 0.0}
        ).text.strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0

    elif dataset.lower() == "logicvista":
        extract_prompt = build_logicvista_extract_prompt(prediction)
        
        # 使用重试机制调用模型
        extracted_answer = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=extract_prompt,
            generation_config={"temperature": 0.0}
        ).text.strip().upper()
        
         # score_prompt = build_score_prompt(question, extracted_answer, answer)
        score_prompt = build_logicvista_score_prompt(question, extracted_answer, answer)
        
        # 直接调用生成内容，依靠外部重试逻辑
        response_text = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=score_prompt,
            generation_config={"temperature": 0.0}
        ).text.strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0
    
    elif dataset.lower() == "r1_onevision_bench":
        extract_prompt = build_r1_onevision_extract_prompt(prediction)
        
        # 使用重试机制调用模型
        extracted_answer = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=extract_prompt,
            generation_config={"temperature": 0.0}
        ).text.strip().upper()

        score_prompt = build_r1_onevision_score_prompt(question, extracted_answer, answer)
        
        # 直接调用生成内容，依靠外部重试逻辑
        response_text = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=score_prompt,
            generation_config={"temperature": 0.0}
        ).text.strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0
    
    elif dataset.lower() in ["math12k", "aime24", "math500", "gsm8k", "gpqa", "olympiadbench"]:
        extract_prompt = build_extract_prompt(prediction, question)

        # 使用重试机制调用模型
        extracted_answer = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=extract_prompt,
            generation_config={"temperature": 0.0}
        ).text.strip().upper()
        
        score_prompt = build_score_prompt(question, extracted_answer, answer)

        # 直接调用生成内容，依靠外部重试逻辑
        response_text = retry_with_backoff(
            model.generate_content,
            max_retries=3,
            initial_delay=2,
            contents=score_prompt,
            generation_config={"temperature": 0.0}
        ).text.strip()
        
        if response_text in ['0', '1']:
            return int(response_text)
        return 0.0
    


if  __name__ == "__main__":
    # prompt = build_extract_prompt("PREDICTION", "QUESTION")
    prompt = build_score_prompt("QUESTION", "PREDICTION", "ANSWER")
    print(prompt)