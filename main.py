from fastapi import FastAPI, Query
from pydantic import BaseModel
from openai import AsyncOpenAI
from fastapi.middleware.cors import CORSMiddleware


from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserChoice(BaseModel):
    response_id: str
    selected_option: str


def build_initial_prompt(title: str):
    return f"""
You are an imaginative storyteller. Write the first section of a book titled "{title}".
The story should be engaging and concise (300-500 words max).
At the end, present 3 short "choose your own adventure" options that let the reader guide the next part.

Example format:
[Story content here...]

What do you do next?
1. Option one
2. Option two
3. Option three
"""


def continue_prompt(selected_option: str):
    return f"""
You are continuing a story based on the reader's choice.

The reader chose: "{selected_option}"

Continue the story based on this choice. End the new section with 3 new adventure options.
"""


@app.get("/generate/first-section")
async def generate_first_section(title: str = Query(...)):
    prompt = build_initial_prompt(title)

    response = await client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        temperature=0.8,
        # max_tokens=1000,
    )

    story_text = response.output_text
    return {
        "story": story_text,
        "response_id": response.id,
        "options": [
            option.strip()
            for option in story_text.split("What do you do next?")[-1].splitlines()
            if option.strip().isdigit()
        ],
    }


@app.post("/generate/next-section")
async def generate_next_section(choice: UserChoice):
    prompt = continue_prompt(choice.selected_option)
    response_id = choice.response_id

    response = await client.responses.create(
        model="gpt-4o-mini",
        previous_response_id=response_id,
        input=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    next_story = response.output_text
    return {
        "story": next_story,
        "response_id": response.id,
        "options": [
            option.strip()
            for option in next_story.split("What do you do next?")[-1].splitlines()
            if option.strip().isdigit()
        ],
    }
