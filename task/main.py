def main(data):
    from openai import OpenAI
    from models import free_models
    from api_key import OPENROUTER_API_KEY
    import json

    # Initialize model and API key
    model = free_models[0]
    OPENROUTER_API_KEY = OPENROUTER_API_KEY

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
        3. Ensure extracted phrases are strictly 2-3 words phrases and relevant to the financial context.

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
            - Provide only the extracted positive and negative sentiment phrases in the json format with pharases being strictly 2-3 words, and nothing else.

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

    # Call the API
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    # Get the response
    output = completion.choices[0].message.content

    # Save the output to a file for later use
    with open("analysis_output.txt", "a", encoding="utf-8") as file:
        file.write(output + "\n")
    print("\nOutput saved to 'analysis_output.txt'")

    try:
        data_json = json.loads(output)

        def filter_phrases(phrases):
            # Use a set to ensure uniqueness, then filter based on the word count
            return list({phrase for phrase in phrases if len(phrase.split()) <= 3})

        # Extract positive and negative phrases
        positive_phrases = data_json.get("positive_phrases", [])
        negative_phrases = data_json.get("negative_phrases", [])

        # Apply the filter to enforce 2-3 word phrases
        positive_phrases = filter_phrases(positive_phrases)
        negative_phrases = filter_phrases(negative_phrases)

        # Save the positive phrases to a file
        with open("positive_phrases.txt", "a", encoding="utf-8") as pos_file:
            for phrase in positive_phrases:
                pos_file.write(phrase + "\n")

        print("Positive phrases saved to 'positive_phrases.txt'")

        # Save the negative phrases to a file
        with open("negative_phrases.txt", "a", encoding="utf-8") as neg_file:
            for phrase in negative_phrases:
                neg_file.write(phrase + "\n")

        print("Negative phrases saved to 'negative_phrases.txt'\n")

    except json.JSONDecodeError:
        print("Invalid JSON format\n")


if __name__ == "__main__":
    import json
    import glob
    from tqdm import tqdm

    files = glob.glob('/home/amankumar/prac/imp/sentiment/CallEarningTranscripts/CallEarningTranscripts/working_files/json_files1/processed_json1/processed_*.json')

    for file_path in tqdm(files[:10], leave=False):


        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)  # Parse the JSON content into a Python dictionary or list
        
        for key in data:
            output = data[key]

            if output is not None and output != '':
               main(output)
            else:
                print("data is empty")
            

