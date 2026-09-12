import pandas as pd

low = [
    "I am having a good day",
    "I feel happy today",
    "I enjoyed spending time with my friends",
    "I am excited about the weekend",
    "I feel calm and relaxed",
    "I am looking forward to tomorrow",
    "I had a nice day with my family",
    "I feel positive about my future",
    "Things are going well for me",
    "I am enjoying my studies",
    "I feel comfortable today",
    "I am excited to learn new things",
    "I had a productive day",
    "I feel peaceful this evening",
    "I am enjoying my hobbies",
    "I feel confident about my work",
    "Today was a pleasant day",
    "I feel grateful for my friends",
    "I am feeling motivated",
    "I am enjoying my time with family"
]

medium = [
    "I feel stressed about my exams",
    "I have been feeling worried lately",
    "I am having a difficult day",
    "I feel overwhelmed with my assignments",
    "I am finding it hard to concentrate",
    "I have been feeling lonely recently",
    "I am nervous about my future",
    "I feel stressed because of college",
    "I have too many things to handle",
    "I keep worrying about my problems",
    "I feel frustrated with everything",
    "I am having trouble sleeping because I am worried",
    "I feel mentally tired lately",
    "I am struggling to manage my stress",
    "I feel anxious about my exams",
    "I have been feeling down recently",
    "I feel like everything is becoming difficult",
    "College work is making me stressed",
    "I am worried about my results",
    "I feel pressured by my responsibilities"
]

high = [
    "I feel completely hopeless",
    "I don't know how to deal with this anymore",
    "I feel like I cannot go on",
    "Everything feels unbearable right now",
    "I feel completely helpless",
    "I don't see any way out of this situation",
    "I feel like there is no hope left",
    "I cannot handle this anymore",
    "I feel trapped and hopeless",
    "I feel like giving up on everything",
    "I don't know what to do anymore",
    "I feel completely lost and helpless",
    "I cannot see a way out of my problems",
    "I feel that everything is falling apart",
    "I have lost all hope",
    "I feel extremely helpless right now",
    "I don't think things will ever get better",
    "I feel like there is no way forward",
    "I feel completely overwhelmed and hopeless",
    "I feel unable to cope anymore"
]

data = []

for text in low:
    data.append([text, "low"])

for text in medium:
    data.append([text, "medium"])

for text in high:
    data.append([text, "high"])

df = pd.DataFrame(data, columns=["text", "risk"])

df.to_csv("data/dataset.csv", index=False)

print("Dataset created successfully!")
print("\nNumber of examples:")
print(df["risk"].value_counts())