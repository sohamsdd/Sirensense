import os
import random
from pydub import AudioSegment

BASE_DIR = r"D:\Soham\ADM_Project"
siren_folder = os.path.join(BASE_DIR, "emergency")
traffic_folder = os.path.join(BASE_DIR, "urban")
output_base = os.path.join(BASE_DIR, "Dataset")
os.makedirs(output_base, exist_ok=True)

folders = {
    "emergency": os.path.join(output_base, "emergency"),
    "urban": os.path.join(output_base, "urban"),
    "emergency2urban": os.path.join(output_base, "emergency2urban"),
    "urban2emergency": os.path.join(output_base, "urban2emergency"),
}
for f in folders.values():
    os.makedirs(f, exist_ok=True)

TARGET_DURATION = 5 * 1000  # 5 seconds

def random_clip(folder, duration):
    file = random.choice(os.listdir(folder))
    path = os.path.join(folder, file)
    audio = AudioSegment.from_file(path)
    if len(audio) > duration:
        start = random.randint(0, len(audio) - duration)
        audio = audio[start:start+duration]
    else:
        repeats = (duration // len(audio)) + 1
        audio = (audio * repeats)[:duration]
    return audio

# Save fixed clips
def save_clips(src_folder, dst_folder, count=2500):
    folder_name = os.path.basename(dst_folder.rstrip("/"))
    for i in range(count):
        clip = random_clip(src_folder, TARGET_DURATION)
        clip.export(os.path.join(dst_folder, f"{folder_name}{i}.wav"), format="wav")

save_clips(siren_folder, folders["emergency"])
save_clips(traffic_folder, folders["urban"])

# Mixed datasets
def save_mixed(dst_folder, first, second, count=2500):
    folder_name = os.path.basename(dst_folder.rstrip("/"))
    for i in range(count):
        split = random.randint(1, TARGET_DURATION-1)
        first_clip = random_clip(first, split)
        second_clip = random_clip(second, TARGET_DURATION - split)
        mixed = first_clip + second_clip
        mixed.export(os.path.join(dst_folder, f"{folder_name}{i}.wav"), format="wav")

save_mixed(folders["emergency2urban"], siren_folder, traffic_folder)
save_mixed(folders["urban2emergency"], traffic_folder, siren_folder)

print("✅ Dataset created at:", output_base)


# We had a dataset of around 2,000 .wav files which consisted of pure clips i.e. just emergency or just urban sounds. These audio data samples would range from 3 sec to 15 secs. Using pydub library
# we created 4 classes each of 2,500 data samples i.e. 10,000 data  samples. Data classes Emergency, Urban, Emergency2Urban and Urban2Emergency of 5 seconds. In the 5 sec audio clip the ratio of emergency to urban OR urban to emergency is random. 2-3, 1-4,...

