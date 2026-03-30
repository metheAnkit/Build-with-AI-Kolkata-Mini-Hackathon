# Build-with-AI-Kolkata-Mini-Hackathon

# 🌍 MedVoyage SLM
### Your Personal Travel Health Advisor | Offline | CPU Native | Gemma 3 (270M)

<p align="center">
  <img src="https://img.shields.io/badge/Model-Gemma%203%20270M-blue?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/Built%20with-Smolify.AI-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Runs%20on-CPU%20Only-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Internet-Not%20Required-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Travel%20Health-purple?style=for-the-badge" />
</p>

---

## 🚨 Problem

Every year, thousands of travelers head to the Himalayas, Northeast India, Southeast Asia, or remote destinations completely unprepared. Wrong vaccines. No altitude medicine. No idea what food or water is safe. And when something goes wrong mid trek or mid flight, there is no signal, no doctor nearby, and no reliable guidance.

Travel health advice is either locked behind expensive clinics, buried in generic PDFs, or requires stable internet that simply does not exist in Spiti Valley or rural Meghalaya. The traveler who needs help the most is always the one with the least access to it.

---

## 💡 Solution: MedVoyage SLM

MedVoyage is a Small Language Model fine tuned on Gemma 3 270M that acts as your personal travel health advisor. Completely offline. Runs on everyday CPU. Zero internet needed.

You tell it your destination, trip duration, travel style, and any existing health conditions. It gives you back a complete personalized medical prep pack instantly.

| Output | Description |
|---|---|
| 💉 Vaccines | Required vs recommended for your destination |
| 💊 Medicine Kit | Exact medicines with use cases |
| 🍽️ Food and Water | Safety rules specific to your destination |
| 🌡️ Climate Warnings | Altitude, heat, and humidity health alerts |
| 🏥 Medical Facility | Guidance on arrival |
| 🚨 Red Flags | Emergency symptoms to never ignore |

Built using Smolify's Knowledge Distillation pipeline, frontier AI models acted as the Teacher generating 10,000 high fidelity training examples. The Student model, Gemma 3 270M, learned to reason like a travel medicine specialist. In 60 minutes. On a CPU. For free.

---

## 🧠 Why a Specialized Model Instead of a General LLM?

General LLMs like GPT or Gemini are brilliant at everything but optimized for nothing. They require internet, cost money per query, and cannot run offline on low end devices.

MedVoyage is trained on one thing and one thing only, travel health preparation. That specialization means it is faster, lighter, more accurate for its domain, and works in the exact moment you need it most, when you are already on the road with zero connectivity.

> This is the DeepSeek Effect in action. Not bigger. Smarter. Not renting intelligence. Owning it.

---

## 🎯 Why This Matters

India's outbound travel market crossed 27 million trips last year. Domestic travel crossed 2 billion. Medical emergency claims from travel insurance companies are rising every single year and most of them are completely preventable.

MedVoyage does not replace your doctor. It makes sure you are prepared enough that you never urgently need one mid trek. Small model. Massive impact. Whether you are a solo backpacker heading to Zanskar or a family flying to Bangkok for the first time, MedVoyage fits in your pocket and works where there is no signal. That is not a feature. That is the entire point.

---

## 🚀 Quick Start

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "your-username/medvoyage-slm"

processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

message = [
    {"role": "system", "content": "You are MedVoyage, an expert travel health advisor."},
    {"role": "user", "content": "I am trekking to Leh Ladakh for 10 days. I am 28 years old with no health conditions. What medical prep do I need?"}
]

inputs = processor.apply_chat_template(
    message,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

out = model.generate(
    **inputs.to(model.device),
    pad_token_id=processor.eos_token_id,
    max_new_tokens=256
)

print(processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))
```

---

## 🏗️ How It Was Built

| Component | Details |
|---|---|
| Base Model | Gemma 3 270M |
| Platform | Smolify.AI |
| Method | Knowledge Distillation |
| Dataset Size | 10,000 synthetic examples |
| Training Hardware | L4 GPU via Smolify Pipeline |
| Inference Hardware | CPU only |
| Build Time | 15 minutes |

---

## 📦 Dataset Coverage

| Variable | Values |
|---|---|
| Destination Type | Himalayan, Coastal, Desert, Jungle, Urban, International |
| Trip Duration | Weekend, 1 week, 2 to 3 weeks, 1 month+ |
| Travel Style | Adventure, Pilgrimage, Backpacking, Family, Business |
| Health Profile | Healthy, Diabetic, Asthmatic, Pregnant, Elderly, Child |
| Season | Summer, Monsoon, Winter, Spring |
| Risk Level | Low (Goa), Medium (Assam), High (Leh, Remote Jungle) |

---

## 🙏 Acknowledgements

Built with [Smolify.AI](https://smolify.ai) 🔥

Massive thanks to [Rishiraj Acharya](https://github.com/rishiraj) for building Smolify, a platform that makes Knowledge Distillation accessible to every developer, student, and builder regardless of their resources.

Built at the GDG Cloud Kolkata and ML Kolkata Mini Hackathon.

---

## 🏷️ Tags

`#SmolifyAI` `#GemmaAI` `#KnowledgeDistillation` `#HealthTech` `#TravelHealth` `#AIForGood` `#SLM` `#GDGKolkata` `#MLKolkata` `#EdgeAI` `#CPUAI` `#GoogleGemma` `#BuildWithAI`
