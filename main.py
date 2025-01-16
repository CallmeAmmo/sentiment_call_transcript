import re
from openai import OpenAI
from models import free_models_openrouter, free_models_groq
from api_key import OPENROUTER_API_KEY, GROQ_API_KEY
import json, os, glob, time
from tqdm import tqdm
import shutil

MODEL_OPENROUTER = free_models_openrouter[0]
MODEL_GROQ = free_models_groq[0]

API_KEY_OPENROUTER = OPENROUTER_API_KEY[0]
API_KEY_GROQ = GROQ_API_KEY [1]

BASE_URL_OPENROUTER = "https://openrouter.ai/api/v1"
BASE_URL_GROQ = "https://api.groq.com/openai/v1"


# Initialize model and API key
BASE_URL = BASE_URL_GROQ
API_KEY = API_KEY_GROQ
MODEL = MODEL_GROQ



def create_final_files(file_name):
    # Define paths for final files
    pos_final_path = f"sentiment_files/positive/positive_phrases_{file_name}.txt"
    neg_final_path = f"sentiment_files/negative/negative_phrases_{file_name}.txt"
    analysis_output_path = f"sentiment_files/analysis_output/analysis_output_{file_name}.txt"

    # Ensure the directories exist
    os.makedirs(os.path.dirname(pos_final_path), exist_ok=True)
    os.makedirs(os.path.dirname(neg_final_path), exist_ok=True)
    os.makedirs(os.path.dirname(analysis_output_path), exist_ok=True)


    # Copy data from temporary files to final files
    try:
        # Copy positive phrases
        shutil.copy(f"current_file/positive_phrases_{file_name}_tmp.txt", pos_final_path)
        print(f"Final positive phrases file created: {pos_final_path}")

        # Copy negative phrases
        shutil.copy(f"current_file/negative_phrases_{file_name}_tmp.txt", neg_final_path)
        print(f"Final negative phrases file created: {neg_final_path}")

        # Copy analysis_output phrases
        shutil.copy(f"current_file/analysis_output_{file_name}_tmp.txt", analysis_output_path)
        print(f"Final analysis output file created: {analysis_output_path}")

        # Delete temporary files
        os.remove(f"current_file/positive_phrases_{file_name}_tmp.txt")
        os.remove(f"current_file/negative_phrases_{file_name}_tmp.txt")
        os.remove(f"current_file/analysis_output_{file_name}_tmp.txt")
        os.remove(f"current_file/full_answer.json")

        print("Temporary files deleted.")

    except Exception as e:
        print(f"An error occurred while creating final files or deleting temporary files: {e}")

def call_openai_api(client, model, system_message, user_message, retry_delay=4, max_retries=3):
    """Call the OpenAI API with retry logic for rate-limit errors."""
    retries = 0
    while retries < max_retries:

        # # Call the OpenAI API
        # completion = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": system_message},
        #         {"role": "user", "content": user_message},
        #     ],
        # )

        try:
            # Call the OpenAI API
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            )
        
        
            if completion.id == None :
                
                print(completion)
                print()

                try :
                    raw_metadata = json.loads(completion.error['metadata']['raw'])
                    if 'error' in raw_metadata and raw_metadata['error'].get('type') == 'rate_limit_exceeded':
                        retries += 1
                        print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Retry {retries}/{max_retries})")
                        time.sleep(retry_delay)
                except:
                    print(completion)
            else:
                return completion
            
        except Exception as e:
            print(f"An error occurred while calling the OpenAI API: {e}")
            retries+=1
            time.sleep(retry_delay)
        

    print("Exceeded maximum retries.")
    return None


def filter_phrases(phrases, transcript):
    """Filter out phrases that are not 2-3 words long and ensure uniqueness."""
    return list({phrase for phrase in phrases if (2 <= len(phrase.split()) <= 3) and (phrase in transcript)})


def main(data, file_name):
    # Example list of transcript lines (combined into a single transcript)
    transcript = data

    # Define the task prompt with the entire transcript
    system_message = '''
        You are an advanced assistant skilled in natural language processing and sentiment analysis. Your task is to analyze financial text, specifically earnings call transcripts, and extract strictly meaningful 2-3 word phrases that indicate "positive sentiment" or "negative sentiment."

        The data is from conversations between executives, investors, and analysts, focusing on corporate performance, projections, and investor relations.

        ### Instructions:
        1. Analyze the entire transcript provided below.
        2. For each line in the transcript:
            - Identify positive sentiment phrases: Phrases that convey optimism, growth, confidence, or success.
            - Identify negative sentiment phrases: Phrases that express concerns, risks, doubts, or underperformance.
        3. Ensure extracted phrases are strictly 2-3 word phrases and relevant to the financial context.

        ### Output Format:
        {{
            "positive_phrases": [
                "Positive Sentiment Phrase 1",
                "Positive Sentiment Phrase 2",
                ...
            ],
            "negative_phrases": [
                "Negative Sentiment Phrase 1",
                "Negative Sentiment Phrase 2",
                ...
            ]
        }}

        Important: 
            - Provide only the extracted positive and negative sentiment phrases in the output format with phrases being strictly 2-3 words, and nothing else.

        Here is the full transcript:
        "{transcript}"
    '''

    # Initialize the OpenAI client
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    # Send the entire transcript in one API call
    user_message = system_message.format(transcript=transcript)
    completion = call_openai_api(client, MODEL, system_message, user_message)

    if completion is None:
        return [], []
    
    try:
        # Get the response
        output = completion.choices[0].message.content


        # Save the output to a file for later use
        analysis_output_path_tmp = f"current_file/analysis_output_{file_name}_tmp.txt"
        with open(analysis_output_path_tmp, "a", encoding="utf-8") as file:
            file.write(output + "\n")

    except Exception as e:
        print(f"An error occurred while processing the response: {e}")
        return [], []

    try:
        data_json = json.loads(output)

        # Extract positive and negative phrases
        positive_phrases = filter_phrases(data_json.get("positive_phrases", []), transcript )
        negative_phrases = filter_phrases(data_json.get("negative_phrases", []), transcript)

        # Save the positive phrases to a file
        with open(f"current_file/positive_phrases_{file_name}_tmp.txt", "a", encoding="utf-8") as pos_file:
            pos_file.write("\n".join(positive_phrases) + "\n")

        # Save the negative phrases to a file
        with open(f"current_file/negative_phrases_{file_name}_tmp.txt", "a", encoding="utf-8") as neg_file:
            neg_file.write("\n".join(negative_phrases) + "\n")

        return positive_phrases, negative_phrases

    except json.JSONDecodeError:
        print("Invalid JSON format\n")
        return [], []


if __name__ == "__main__":
    

    files = glob.glob('CallEarningTranscripts/CallEarningTranscripts/working_files/*_output.json')

    for file_path in tqdm(files, leave=False):
        print(file_path)
        file_name = re.split(r'[\\/]', file_path)[-1]
        file_name_abs = file_name.split('.')[0]

        path_to_save = f'sentiment_files/combined/sentiment_{file_name}'

        if os.path.exists(path_to_save):
            print("File exists.")
            continue
        else:

            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            final_answer = {}

            k = 'speaker_texts'
            dialogue = ''
            
            for idx , key in enumerate(data[k]):
                dialogue += key['dialogue']

                if dialogue and (idx%5==0 or idx==len(data[k])-1):
                    positive_phrases, negative_phrases = main(dialogue, file_name_abs)

                    ans = {
                        'positive_phrases': positive_phrases,
                        'negative_phrases': negative_phrases
                    }

                    final_answer[dialogue] = ans

                    with open('current_file/full_answer.json', "a") as json_file:
                        json.dump({dialogue: ans}, json_file, indent=4)
                        json_file.write("\n")
                    
                    dialogue =''
                else:
                    # print("Data is empty")
                    pass

        with open(path_to_save, "w") as json_file:
            json.dump(final_answer, json_file, indent=4)
        
        create_final_files(file_name_abs)
        
        with open('processed_files.txt', "a") as json_file:
            json_file.write(file_name + "\n")

