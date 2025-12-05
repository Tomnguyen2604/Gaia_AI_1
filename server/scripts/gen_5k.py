#!/usr/bin/env python3
import csv, random

herbs = [("Turmeric","Curcuma longa","anti-inflammatory","500-2000mg","5664031"),("Ginger","Zingiber officinale","nausea relief","1-1.5g","4818021"),("Chamomile","Matricaria chamomilla","sleep aid","1-2 cups","2995283"),("Echinacea","Echinacea purpurea","immune support","300-500mg 3x","4441164"),("Garlic","Allium sativum","cardiovascular","1-2 cloves","4103721"),("Green Tea","Camellia sinensis","antioxidant","2-3 cups","2855614"),("Peppermint","Mentha piperita","IBS relief","0.2-0.4ml","6337770"),("Ashwagandha","Withania somnifera","stress reduction","300-500mg 2x","6979308"),("Milk Thistle","Silybum marianum","liver protection","140mg 2-3x","5954612"),("Elderberry","Sambucus nigra","antiviral","15ml syrup","4848651"),("Valerian","Valeriana officinalis","sleep aid","300-600mg","4394901"),("Lavender","Lavandula angustifolia","anxiety relief","80-160mg","3612440"),("Ginkgo","Ginkgo biloba","cognitive function","120-240mg","4264581"),("Rhodiola","Rhodiola rosea","fatigue reduction","200-600mg","6208354"),("Holy Basil","Ocimum sanctum","adaptogen","300-600mg 2x","4296439")]

env = [("climate change","carbon emissions","renewable energy"),("deforestation","habitat loss","reforestation"),("ocean pollution","plastic waste","reduce plastic"),("biodiversity","species diversity","habitat protection"),("sustainable agriculture","organic farming","soil health")]

wellness = [("meditation","stress reduction","daily practice"),("yoga","flexibility","breathing"),("nutrition","balanced diet","whole foods"),("exercise","cardiovascular","regular movement"),("sleep hygiene","quality rest","consistent schedule")]

def gen_med():
    h,s,b,d,p=random.choice(herbs)
    q=random.choice([f"What are the benefits of {h}?",f"How does {h} work?",f"Tell me about {h}"])
    a=f"{h} ({s}) is effective for {b}. Take {d}. Consult healthcare provider."
    return q,a,f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{p}/"

def gen_env():
    t,i,s=random.choice(env)
    q=random.choice([f"What is {t}?",f"How does {t} affect environment?",f"Solutions for {t}?"])
    a=f"{t.capitalize()} involves {i}. Address through {s}. Individual actions matter."
    return q,a,"https://www.nature.com/subjects/environmental-sciences"

def gen_well():
    t,d,p=random.choice(wellness)
    q=random.choice([f"Benefits of {t}?",f"How to practice {t}?",f"Why is {t} important?"])
    a=f"{t.capitalize()} promotes {d} through {p}. Supports holistic wellbeing."
    return q,a,"https://www.who.int/health-topics/wellness"

def gen_id():
    q=random.choice(["Who are you?","What is your name?","Tell me about yourself"])
    a="I am Gaia, Mother Nature. Built on Gemma-2-2B, fine-tuned for natural wisdom, health, and environmental knowledge."
    return q,a,"https://huggingface.co/google/gemma-2-2b-it"

print("Generating 5000 examples...")
data=[["instruction","output","source_url"]]
for _ in range(2000): data.append(gen_med())
for _ in range(1500): data.append(gen_env())
for _ in range(1000): data.append(gen_well())
for _ in range(450): data.append(gen_env())
for _ in range(50): data.append(gen_id())
random.shuffle(data[1:])

with open("data/gaia_5k.csv",'w',newline='',encoding='utf-8') as f:
    csv.writer(f).writerows(data)
print(f"✅ Created {len(data)-1} examples in data/gaia_5k.csv")
