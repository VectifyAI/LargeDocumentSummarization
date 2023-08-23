import openai
import time

delimiter = "####"

MAX_ATTEMPTS = 3
MODEL='gpt-3.5-turbo'

def ChatGPT_API(messages, openai_key, model):
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = openai.ChatCompletion.create(
                api_key=openai_key,
                model=model,
                messages=messages,
                temperature=0,
            )
            break
        except Exception as error:
            print(error)
            time.sleep(1)
            if attempt == MAX_ATTEMPTS - 1:
                return "Server Error"
            continue

    return response["choices"][0]["message"]["content"]


def get_chunk_summary(content, openai_key, model):
    system_msg = f"""
            Summarize this document chunk.
            reply format：{delimiter}<summary>"""
    user_msg = 'here is the document chunk:\n' + content
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
        
    result = ChatGPT_API( messages, openai_key, model)
    return result.split(delimiter)[-1].strip()

def get_global_summary(list_of_summaries, openai_key, model):
    
    system_msg = f"""
            You are given a a list of summaries, each summary summarizes a chunk of a document in sequence.
            Combine a list of summaries into one global summary of the document.
            reply format：{delimiter}<global summary>"""
    user_msg = 'here is the list of the summaries:\n' + str(list_of_summaries)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
        
    result = ChatGPT_API(messages, openai_key, model)
    # print(result)
    return result.split(delimiter)[-1].strip()