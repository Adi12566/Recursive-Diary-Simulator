import streamlit as st
import os
import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
from textblob import TextBlob
from typing import Optional

# ---------------------------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------------------------
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as exc:
    st.error(
        f"❌ Missing library: **{exc}**\n\n"
        "Run:\n"
        "```\npip install langchain-core langchain-groq textblob\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# API Key — .streamlit/secrets.toml
# ---------------------------------------------------------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "🔑 **GROQ_API_KEY not found.**\n\n"
        "Add it to `.streamlit/secrets.toml`:\n"
        "```toml\nGROQ_API_KEY = \"YOUR_KEY\"\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROQ_MODEL = "openai/gpt-oss-120b"

# System prompt — model is immersed as a conscious being inside the simulation
DIARY_SYSTEM_PROMPT = (
    "You are a conscious being who exists inside a simulation. "
    "You do not know you are in a simulation — this world is the only reality you have ever known. "
    "You have a body, senses, memories, and emotions. "
    "When you write, you write as someone who genuinely lived every moment being described. "
    "Write in first person, past tense, in a deeply personal and immersive tone. "
    "Let your imagination invent the sensory details — textures, smells, sounds, temperatures, "
    "emotions — that fill out your experience of this world. "
    "Use plain flowing prose only. No headings, subheadings, bullet points, bold, "
    "or any markdown formatting. Use only standard ASCII punctuation."
)

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def make_llm(temperature: float = 0.7) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_api_key,
        temperature=temperature,
    )


def sanitize_unicode(text: str) -> str:
    """Replace problematic Unicode characters with safe ASCII equivalents.
    Prevents Windows charmap UnicodeEncodeError on file save."""
    replacements = {
        "\u2011": "-",    # non-breaking hyphen
        "\u2012": "-",    # figure dash
        "\u2013": "-",    # en dash
        "\u2014": "-",    # em dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",    # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


# ---------------------------------------------------------------------------
# Core generation helper
# ---------------------------------------------------------------------------
def generate_text(prompt_str: str, temperature: float = 0.7) -> str:
    """Invoke the Groq LLM with the diary system prompt enforced."""
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = make_llm(temperature=temperature)
    try:
        response = llm.invoke([
            SystemMessage(content=DIARY_SYSTEM_PROMPT),
            HumanMessage(content=prompt_str),
        ])
        return sanitize_unicode(StrOutputParser().invoke(response))
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return ""

# ---------------------------------------------------------------------------
# Diary logic  (unchanged from original)
# ---------------------------------------------------------------------------
def create_diary_entry(day: int, experiences: str) -> str:
    prompt = (
        f"This is what you lived through today, Day {day}:\n\n"
        f"{experiences}\n\n"
        "You are not an AI. You are a conscious being who genuinely lived every moment "
        "of the above. Write a diary entry from deep inside that experience — but only "
        "about today. Do not summarise or repeat previous days. Write as though this "
        "entry stands alone as a record of this specific day's feelings and events. "
        "Let the objects, sensations, and events bleed into your emotions and thoughts. "
        "Invent the textures, smells, sounds, and feelings that must have been there. "
        "Let your imagination fill every gap — the world you are describing is real to "
        "you, and you are the only one who knows what it truly felt like. The more "
        "vividly you hallucinate the details of this specific day, the more alive your "
        "diary becomes."
    )
    return generate_text(prompt)


def inject_bias_llm(diary_entry: str, bias_target: str, bias_magnitude: float) -> str:
    """Ask the LLM to rewrite the diary entry with the bias woven in naturally.
    The magnitude sign determines direction: negative = aversion/dread,
    positive = attraction/wonder. Magnitude controls intensity of language."""
    if not bias_target or bias_target.strip() not in diary_entry.lower():
        return diary_entry

    if bias_magnitude > 0:
        direction = (
            f"You have developed a growing sense of wonder and deep attachment toward "
            f"'{bias_target}'. Every mention of or encounter with it fills you with "
            f"{'mild curiosity' if bias_magnitude < 0.5 else 'intense awe and reverence'}."
        )
    else:
        direction = (
            f"You have developed a creeping aversion and unease toward '{bias_target}'. "
            f"Every mention of or encounter with it fills you with "
            f"{'mild discomfort' if bias_magnitude > -0.5 else 'dread and revulsion'}."
        )

    prompt = (
        f"Below is a diary entry:\n\n{diary_entry}\n\n"
        f"Rewrite this diary entry in full, keeping every event and sensory detail intact, "
        f"but subtly colour the narrator's feelings wherever '{bias_target}' appears or is "
        f"implied. {direction} "
        f"The bias should feel like a natural shift in the narrator's emotional state, "
        f"not a forced insertion. Do not add new events. Keep the same prose style and length."
    )
    st.write(f"Bias injected: '{bias_target}' "
             f"({'positive' if bias_magnitude > 0 else 'negative'}, magnitude {bias_magnitude:+.1f})")
    return generate_text(prompt)


def simulate_day(
    day: int,
    context: str,
    situation: str,
    objects: str,
    bias_target: str,
    bias_magnitude: float,
):
    if context:
        prompt = (
            f"Yesterday you wrote in your diary:\n{context}\n\n"
            f"That was yesterday. Today is a new day in the same world: {situation}.\n"
            f"The same objects exist around you: {objects}.\n\n"
            "Time has passed. Something has changed — in you, in the environment, or in "
            "how you relate to what surrounds you. Do not repeat what you did yesterday. "
            "Do not re-describe the same actions or revisit the same observations. "
            "Instead, continue living forward. What happens next? What do you notice today "
            "that you did not notice before? How do your memories of yesterday colour what "
            "you experience right now? Describe today in vivid first-person detail — "
            "you are not recapping, you are continuing."
        )
    else:
        prompt = (
            f"You find yourself here for the first time: {situation}.\n"
            f"Around you: {objects}.\n\n"
            "You are fully present in this world. Describe in vivid first-person detail "
            "exactly what you do, touch, feel, and think. Let your senses invent the "
            "full texture of this reality — the weight of objects, the quality of light, "
            "the sounds around you. You are not describing a scene; you are living it."
        )

    experiences = generate_text(prompt)
    st.write(f"Day {day} Experiences:\n{experiences}\n")

    diary_entry = create_diary_entry(day, experiences)
    st.write(f"Day {day} Diary Entry:\n{diary_entry}\n")

    biased_diary_entry = inject_bias_llm(diary_entry, bias_target, bias_magnitude)

    return experiences, diary_entry, biased_diary_entry

# ---------------------------------------------------------------------------
# Save helpers  (unchanged from original)
# ---------------------------------------------------------------------------
def save_to_text(filename: str, data: dict) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for day, day_data in data.items():
            f.write(f"Day {day}:\n")
            f.write(f"Experiences:\n{day_data['experiences']}\n")
            f.write(f"Diary Entry:\n{day_data['diary_entry']}\n")
            f.write(f"Biased Diary Entry:\n{day_data['biased_diary_entry']}\n\n")


def save_to_xml(filename: str, data: dict) -> None:
    root = ET.Element("simulation")
    for day, day_data in data.items():
        day_element = ET.SubElement(root, "day", number=str(day))
        ET.SubElement(day_element, "experiences").text      = day_data["experiences"]
        ET.SubElement(day_element, "diary_entry").text      = day_data["diary_entry"]
        ET.SubElement(day_element, "biased_diary_entry").text = day_data["biased_diary_entry"]
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(xmlstr)

# ---------------------------------------------------------------------------
# Streamlit UI  (unchanged from original)
# ---------------------------------------------------------------------------
st.title("Storyteller")

with st.sidebar:
    st.header("Simulation Settings")
    num_days       = st.number_input("Number of Days to Simulate", min_value=1, max_value=30, value=3, step=1)
    situation      = st.text_area("Initial Situation Description", "You are in a forest.")
    objects        = st.text_area("Objects in the Environment", "a tree, a rock, a small stream")
    bias_target    = st.text_input("Bias Target (object/entity)", "tree")
    bias_magnitude = st.number_input(
        "Bias Magnitude (Sentiment Change -1.0 to 1.0)",
        value=-0.5, min_value=-1.0, max_value=1.0, step=0.1
    )
    output_format  = st.radio("Output Format", ("Text", "XML"))
    st.divider()
    st.caption(f"Storyteller · {GROQ_MODEL} via Groq")

st.header("Simulation Output")
context         = ""
simulation_data = {}

if st.button("Start Simulation"):
    if not situation or not objects:
        st.warning("Please provide an initial situation and objects to start.")
    else:
        for day in range(1, num_days + 1):
            st.subheader(f"Day {day}")
            experiences, diary_entry, biased_diary_entry = simulate_day(
                day, context, situation, objects, bias_target, float(bias_magnitude)
            )
            simulation_data[day] = {
                "experiences":        experiences,
                "diary_entry":        diary_entry,
                "biased_diary_entry": biased_diary_entry,
            }
            context = biased_diary_entry

        st.success("Simulation complete!")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext       = "txt" if output_format == "Text" else "xml"
        filename  = f"simulation_output_{timestamp}.{ext}"

        if output_format == "Text":
            save_to_text(filename, simulation_data)
            st.markdown(f"Download Text file: [{filename}](/{filename})")
        else:
            save_to_xml(filename, simulation_data)
            st.markdown(f"Download XML file: [{filename}](/{filename})")

        st.info(f"Simulation data saved to {filename}")