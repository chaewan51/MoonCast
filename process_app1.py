import os
import glob
import json
import base64
import io

from pydub import AudioSegment
from inference import Model

# --- your English reference audio/text mapping ---
EN_REF_AUDIO = {
    "0": {
        "ref_audio": "./en_prompt0.wav",
        "ref_text": "Yeah, no, this is my backyard. It's never ending so just the way I like it. So social distancing has never been a problem.",
    },
    "1": {
        "ref_audio": "./en_prompt1.wav",
        "ref_text": "I'm doing great and, look, it couldn't be any better than having you at your set, which is the outdoors.",
    }
}

def sanitize_dialogue(raw):
    """
    Merge consecutive turns with same speaker (role),
    but DO NOT drop last turn even if odd.
    """
    merged = []
    for turn in raw:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["text"] += " " + turn["text"]
        else:
            merged.append(turn)

    # ensure no two consecutive roles are the same
    roles = [t["role"] for t in merged]
    for i in range(len(roles) - 1):
        if roles[i] == roles[i + 1]:
            raise ValueError(f"Speaker roles not alternating at {i}/{i+1}: {roles[i]}/{roles[i+1]}")
    print("Final roles order:", roles)
    return merged

def main():
    input_folder = "claimJson"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    model = Model()
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    if not json_files:
        print(f"No JSON files found in '{input_folder}'")
        return

    for path in json_files:
        fname = os.path.splitext(os.path.basename(path))[0]

        out_path = os.path.join(output_folder, f"{fname}.wav")
        if os.path.exists(out_path):
            print(f"✓ Skipping {fname} (already exists)")
            continue

        print(f"→ Processing {fname} …", end=" ")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            raw = data["dialogue"] if isinstance(data, dict) else data
            cleaned = sanitize_dialogue(raw)
            js = {"dialogue": cleaned, "role_mapping": EN_REF_AUDIO}

            audio_b64 = model.inference(js)
            audio_bytes = base64.b64decode(audio_b64)

            # read MP3 and pad front/back to avoid clipping
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            pad = AudioSegment.silent(duration=300)  # 0.3s
            audio = pad + audio + pad

            audio.export(out_path, format="wav")
            print(f"✓ Saved to {out_path}")

        except Exception as e:
            print(f"❌ Failed {fname}: {e}")

if __name__ == "__main__":
    main()
