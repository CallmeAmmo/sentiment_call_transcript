
from openai import OpenAI
from models import free_models
from api_key import OPENROUTER_API_KEY
import json, os, glob, time
# import glob
from tqdm import tqdm
# import time

# Initialize model and API key
MODEL = free_models[0]
OPENROUTER_API_KEY = OPENROUTER_API_KEY[1]


def call_openai_api(client, model, system_message, user_message, retry_delay=3, max_retries=3):
    """Call the OpenAI API with retry logic for rate-limit errors."""
    retries = 0
    while retries < max_retries:
        try:
            # Call the OpenAI API
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            )
            return completion
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                retries += 1
                print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Retry {retries}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"An unexpected error occurred: {e}")
                return None
    print("Exceeded maximum retries.")
    return None


def filter_phrases(phrases, transcript):
    """Filter out phrases that are not 2-3 words long and ensure uniqueness."""
    return list({phrase for phrase in phrases if (2 <= len(phrase.split()) <= 3) and (phrase in transcript)})


def main(data):
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
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
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
        with open("analysis_output.txt", "a", encoding="utf-8") as file:
            file.write(output + "\n")
        print("\nOutput saved to 'analysis_output.txt'")
    except Exception as e:
        print(f"An error occurred while processing the response: {e}")
        print(completion)
        return [], []

    try:
        data_json = json.loads(output)

        # Extract positive and negative phrases
        positive_phrases = filter_phrases(data_json.get("positive_phrases", []), transcript )
        negative_phrases = filter_phrases(data_json.get("negative_phrases", []), transcript)

        # Save the positive phrases to a file
        with open("positive_phrases.txt", "a", encoding="utf-8") as pos_file:
            pos_file.write("\n".join(positive_phrases) + "\n")
        print("Positive phrases saved to 'positive_phrases.txt'")

        # Save the negative phrases to a file
        with open("negative_phrases.txt", "a", encoding="utf-8") as neg_file:
            neg_file.write("\n".join(negative_phrases) + "\n")
        print("Negative phrases saved to 'negative_phrases.txt'\n")

        return positive_phrases, negative_phrases

    except json.JSONDecodeError:
        print("Invalid JSON format\n")
        return [], []


if __name__ == "__main__":
    

    files = glob.glob('/home/amankumar/prac/imp/sentiment/CallEarningTranscripts/CallEarningTranscripts/working_files/*_output.json')


    

    for file_path in tqdm(files[:1], leave=False):
        print(file_path)
        file_name = file_path.split('/')[-1]


        path_to_save = f'/home/amankumar/prac/imp/sentiment/sentiment_files/all/sentiment_{file_name}'

        if os.path.exists(path_to_save):
            print("File exists.")
        else:

            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            final_answer = {}

            k = 'speaker_texts'
            dialogue = ''
            
            for idx , key in enumerate(data[k]):
                dialogue += key['dialogue']

    # -------------------------------------------------
            # print(dialogue)

                if dialogue and (idx%7==0 or idx==len(data[k])-1):
                    positive_phrases, negative_phrases = main(dialogue)

                    ans = {
                        'positive_phrases': positive_phrases,
                        'negative_phrases': negative_phrases
                    }

                    final_answer[dialogue] = ans

                    with open('full_answer.json', "a") as json_file:
                        json_file.write("\n")
                        json.dump({dialogue: ans}, json_file, indent=4)
                        json_file.write("\n")
                    
                    dialogue =''
                else:
                    # print("Data is empty")
                    pass

    # -------------------------------------------------
                    

        with open('full_answer_complete.json', "w") as json_file:
            json.dump(final_answer, json_file, indent=4)

