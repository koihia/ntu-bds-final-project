import base64
import tempfile
import cv2
import openai
import streamlit as st
import supervision as sv


def check_openai_api_key(client: openai.OpenAI) -> bool:
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


"""
# Automated Video Narration with GPT Vision and TTS
"""

openai_api_key = st.text_input("OpenAI API Key", type="password")
client = openai.OpenAI(api_key=openai_api_key)
if not openai_api_key:
    "Please enter OpenAI API key to continue."
    st.stop()
if not check_openai_api_key(client):
    st.error("Invalid api key.")
    st.stop()

"""
## Upload Video
"""

uploaded_file = st.file_uploader("Choose a video file")

if uploaded_file is None:
    st.stop()

video_file = tempfile.NamedTemporaryFile()
video_file.write(uploaded_file.read())
video_path = video_file.name

f"### Filename: {uploaded_file.name}"

st.video(uploaded_file)

"""
## Narration
"""

if not st.button("Generate Narration", type="primary"):
    st.stop()

video_info = sv.VideoInfo.from_video_path(video_path=video_path)

FRAME_EXTRACTION_FREQUENCY_SECONDS = 2

PROMPT = (
    f"The uploaded series of images is from a single video. "
    f"The frames were sampled every {FRAME_EXTRACTION_FREQUENCY_SECONDS} seconds. "
    f"Make sure it takes about {FRAME_EXTRACTION_FREQUENCY_SECONDS // 2} seconds to voice the description of each frame. "
    f"Use exclamation points and capital letters to express excitement if necessary. "
    f"Describe the video using David Attenborough style."
)

frame_extraction_frequency = FRAME_EXTRACTION_FREQUENCY_SECONDS * video_info.fps
frame_generator = sv.get_video_frames_generator(
    source_path=video_path, stride=frame_extraction_frequency
)
base64_frames = []
for frame in frame_generator:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        st.error("Could not encode image to JPEG format.")
        st.stop()
    base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT,
            },
            *map(
                lambda x: {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{x}"},
                },
                base64_frames,
            ),
        ],
    },
]
params = {
    "model": "gpt-4-vision-preview",
    "messages": messages,
    "max_tokens": 500,
}

narration_pending_message = st.empty()
narration_pending_message.text("Generating narration...")
narration_text = """
In the vast white canvas of the Arctic, a drama unfolds. A pack of grey wolves closely tails a bison, whose steamy breaths betray its exertion. The wolves expertly navigate the snow, their survival depending on teamwork and strategy. Suddenly, they close in, their tawny fur contrasting starkly against the snow. The bison, a lone fortress, holds steadfast, muscles rippling, ready to defend its life.

Despite its size and strength, the wolves are relentless. With each passing moment, the tension escalates - it's a battle of endurance. A spray of snow erupts as the bison charges, attempting to break the relentless siege. The wolves respond with agility and cunning, circling their formidable adversary.

The bison, now tiring, is a testament to the harsh realities of nature's way. The wolves close in further, their eyes fixed, their bodies in perfect synchrony. In a sudden burst of movement, the bison falters, an indication that the scales are tipping. The pack, sensing victory, tightens the noose. 

And in the end, it is nature's law that prevails. The wolves have claimed their hard-won prize, a crucial lifeline in this unforgiving landscape. As the sun casts long shadows upon the scene, the cycle of survival continues, unabridged and unyielding.
"""

result = client.chat.completions.create(**params)
narration_text = result.choices[0].message.content
narration_pending_message.text("Done!")

st.code(narration_text)

"""
## TTS
"""

voice_pending_message = st.empty()
voice_pending_message.text("Generating voice...")

speech_file = tempfile.NamedTemporaryFile()
response = client.audio.speech.create(
    model="tts-1-hd", voice="alloy", input=narration_text
)
speech_file.write(response.read())

voice_pending_message.text("Done!")
st.audio(speech_file.name)
