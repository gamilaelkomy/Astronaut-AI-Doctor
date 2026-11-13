"""AETHER-Care prototype: Astronaut AI Doctor (simulation only). Run: streamlit run aether_app.py"""
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

# ======================= SENSORS (simulated) =======================
@dataclass
class Reading:
    hr: float      # heart rate (bpm)            <- radar / MAX86150
    hrv: float     # HRV RMSSD (ms)              <- radar / MAX86150
    temp: float    # skin/core temp (C)          <- thermal cam / MLX90614
    eda: float     # sweat (uS)                  <- GSR sensor
    resp: float    # breathing (breaths/min)     <- radar
    spo2: float    # blood oxygen (%)            <- MAX86150 / MAX30102
    co2: float     # cabin / helmet CO2 (ppm)    <- BME688 / SCD40
    motion: float  # 0 = still, 10 = heavy effort <- IMU (BMI270)
    ts: float = 0.0

    def as_dict(self):
        return asdict(self)


SCENARIOS = {
    "Resting":   dict(hr=72,  hrv=48, temp=36.7, eda=2.2, resp=14, spo2=98, co2=800,  motion=1.5),
    "Exercise":  dict(hr=125, hrv=25, temp=37.2, eda=5.0, resp=24, spo2=97, co2=1100, motion=8),
    "Panic (EVA)": dict(hr=128, hrv=18, temp=36.9, eda=8.0, resp=26, spo2=97, co2=900, motion=2.5),
    "CO2 leak":  dict(hr=92,  hrv=40, temp=36.8, eda=2.8, resp=20, spo2=95, co2=6000, motion=2),
    "Emergency": dict(hr=165, hrv=12, temp=38.3, eda=9.0, resp=30, spo2=86, co2=900,  motion=0),
}


# ======================= ENGINE =======================
FEATURES = ["hr", "hrv", "temp", "eda", "resp"]

# PROTOTYPE thresholds. Replace with values validated by NASA standards / flight surgeons.
T = dict(
    hr_yellow=110, hr_red=150, ex_hr_yellow=150, ex_hr_red=175,
    temp_yellow=37.6, temp_red=38.0, ex_temp_yellow=38.0, ex_temp_red=38.5,
    resp_yellow=22, resp_red=28, ex_resp_yellow=32, ex_resp_red=38,
    spo2_yellow=94, spo2_red=88,
    co2_yellow=2000, co2_red=5000,
    exertion_motion=6, still_motion=1,
)

PHYS = {
    "en": {
        "no_motion": "No movement with abnormal vitals. If you can hear me, move your hand or respond.",
        "hr_critical": "Your heart rate is critical. Sit down now and stop all effort. Breathe in for 4 seconds, out for 6. Turn on cooling.",
        "hr_high": "Your heart rate is high. Reduce effort and drink water.",
        "temp_high": "Your temperature is high. Stop effort, cool down, and drink water.",
        "temp_warm": "Your temperature is a little above normal. Ease off and keep an eye on it.",
        "resp_high": "Your breathing is very fast. Sit down, breathe slowly, and alert your crewmate.",
        "resp_fast": "Your breathing is faster than usual. Slow down and breathe slowly.",
        "spo2_low": "Blood oxygen is low. Check oxygen supply and ventilation now and alert your crewmate.",
        "spo2_border": "Blood oxygen is slightly low. Breathe slowly and check ventilation.",
        "co2_high": "CO2 is dangerously high. Check ventilation or helmet airflow now and alert your crewmate.",
        "co2_elevated": "CO2 is above normal. Check ventilation and take slow breaths.",
        "unusual": "Your readings are different from your usual. Rest and keep an eye on yourself.",
        "recovering": "Readings are returning to normal. Keep resting, monitoring continues.",
    },
    "ar": {
        "no_motion": "لا توجد حركة مع قراءات حيوية غير طبيعية. لو تسمعني، حرّك يدك أو ردّ عليّ.",
        "hr_critical": "نبضك مرتفع جدا. اجلس فورا وأوقف أي مجهود. تنفس ببطء: شهيق 4 ثوان وزفير 6. شغّل التبريد.",
        "hr_high": "نبضك مرتفع. خفف المجهود واشرب ماء.",
        "temp_high": "حرارتك مرتفعة. أوقف المجهود وبرّد جسمك واشرب ماء.",
        "temp_warm": "حرارتك أعلى قليلا من المعتاد. خفف المجهود وراقبها.",
        "resp_high": "تنفسك سريع جدا. اجلس وتنفس ببطء وأبلغ زميلك.",
        "resp_fast": "تنفسك أسرع من المعتاد. تمهّل وتنفس ببطء.",
        "spo2_low": "نسبة الأكسجين في الدم منخفضة. تحقق من إمداد الأكسجين والتهوية الآن وأبلغ زميلك.",
        "spo2_border": "نسبة الأكسجين أقل قليلا من المعتاد. تنفس ببطء وتحقق من التهوية.",
        "co2_high": "ثاني أكسيد الكربون مرتفع بشكل خطير. افحص التهوية أو تدفق الهواء في الخوذة الآن وأبلغ زميلك.",
        "co2_elevated": "ثاني أكسيد الكربون أعلى من المعتاد. افحص التهوية وتنفس ببطء.",
        "unusual": "قراءاتك تختلف عن المعتاد. استرح وراقب نفسك.",
        "recovering": "القراءات تعود إلى المعدل الطبيعي. واصل الراحة والمراقبة مستمرة.",
    },
}
MENTAL = {
    "en": {
        "sensor_med": "Stress indicators detected. Take a 5 minute break and breathe slowly: in 4, out 6.",
        "sensor_high": "Stress indicators are high. Sit somewhere quiet and breathe slowly. Talk to a crewmate or the support assistant.",
        "panic_pattern": "This pattern looks like an acute stress episode, not physical effort. Focus on your breathing: in 4, out 6, and name five things you can see.",
        "form_bad": "Your daily check-in suggests low sleep or low mood. Lighten your workload if you can and talk to a crewmate.",
    },
    "ar": {
        "sensor_med": "هناك مؤشرات توتر. خذ استراحة 5 دقائق وتنفس ببطء: شهيق 4 وزفير 6.",
        "sensor_high": "مؤشرات التوتر عالية. اجلس في مكان هادئ وتنفس ببطء. تحدث مع زميلك أو مع مساعد الدعم.",
        "panic_pattern": "النمط الحالي يشبه نوبة قلق حادة وليس مجهودا بدنيا. ركّز على تنفسك: شهيق 4 وزفير 6، وسمِّ خمسة أشياء تراها حولك.",
        "form_bad": "إجاباتك اليوم تشير إلى قلة نوم أو مزاج منخفض. خفف الحمل إن أمكن وتحدث مع أحد من الطاقم.",
    },
}
CREW = {
    "en": {"red": "Alert to the crew. Your crewmate needs help immediately.",
           "stress": "Alert to the crew. Your crewmate is under stress."},
    "ar": {"red": "تنبيه للطاقم. زميلكم يحتاج مساعدة فورا.",
           "stress": "تنبيه للطاقم. زميلكم تحت ضغط."},
}
IMMEDIATE = {"spo2_low", "co2_high", "no_motion"}  # life-threatening: skip debounce


@dataclass
class Assessment:
    phys_level: int = 0
    ment_level: int = 0
    phys_reasons: list = field(default_factory=list)
    ment_reasons: list = field(default_factory=list)
    stress: float = 0.0
    exertion: bool = False
    anomaly: bool = False
    vitals: dict = field(default_factory=dict)

    def messages(self, lang="en"):
        return [PHYS[lang][r] for r in self.phys_reasons] + [MENTAL[lang][r] for r in self.ment_reasons]

    def spoken(self, lang="en"):
        text = " ".join(self.messages(lang))
        if not text:
            return ""
        if "no_motion" in self.phys_reasons or self.phys_level == 2:
            text += " " + CREW[lang]["red"]
        elif self.ment_level >= 1:
            text += " " + CREW[lang]["stress"]
        return text


def make_baseline(n=600, seed=42):
    """Simulated 10-minute calm baseline. On real hardware, replace with the astronaut's own first 10 minutes."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "hr": rng.normal(70, 3, n), "hrv": rng.normal(50, 4, n), "temp": rng.normal(36.7, 0.1, n),
        "eda": rng.normal(2, 0.4, n), "resp": rng.normal(14, 1, n),
    })


class _Debounce:
    """Escalate after `confirm` consecutive samples, de-escalate after `clear`. Stops flickering."""

    def __init__(self, confirm=2, clear=4):
        self.confirm, self.clear = confirm, clear
        self.level = 0
        self.up = self.down = 0

    def step(self, raw, immediate=False):
        if raw > self.level:
            self.up += 1
            self.down = 0
            if immediate or self.up >= self.confirm:
                self.level, self.up = raw, 0
        elif raw < self.level:
            self.down += 1
            self.up = 0
            if self.down >= self.clear:
                self.level, self.down = raw, 0
        else:
            self.up = self.down = 0
        return self.level


class Monitor:
    def __init__(self, baseline=None, window=1, confirm=1, clear=1):
        base = baseline if baseline is not None else make_baseline()
        self.model = IsolationForest(contamination=0.01, random_state=42).fit(base[FEATURES])
        self.bmean, self.bstd = base[FEATURES].mean(), base[FEATURES].std()
        self.win = deque(maxlen=window)
        self.dp, self.dm = _Debounce(confirm, clear), _Debounce(confirm, clear)

    # ---- helpers
    def stress_score(self, v):
        z = [(self.bmean["hrv"] - v["hrv"]) / self.bstd["hrv"],
             (v["eda"] - self.bmean["eda"]) / self.bstd["eda"],
             (v["resp"] - self.bmean["resp"]) / self.bstd["resp"]]
        return sum(min(max(x, 0), 4) for x in z) / 3

    @staticmethod
    def _phys_raw(v, exertion, anomaly):
        ex = "ex_" if exertion else ""
        r = []
        if v["motion"] <= T["still_motion"] and (v["hr"] > 130 or v["hr"] < 50):
            r.append(("no_motion", 2))
        if v["hr"] > T[ex + "hr_red"]:
            r.append(("hr_critical", 2))
        elif v["hr"] > T[ex + "hr_yellow"]:
            r.append(("hr_high", 1))
        if v["temp"] >= T[ex + "temp_red"]:
            r.append(("temp_high", 2))
        elif v["temp"] >= T[ex + "temp_yellow"]:
            r.append(("temp_warm", 1))
        if v["resp"] >= T[ex + "resp_red"]:
            r.append(("resp_high", 2))
        elif v["resp"] >= T[ex + "resp_yellow"]:
            r.append(("resp_fast", 1))
        if v["spo2"] < T["spo2_red"]:
            r.append(("spo2_low", 2))
        elif v["spo2"] < T["spo2_yellow"]:
            r.append(("spo2_border", 1))
        if v["co2"] >= T["co2_red"]:
            r.append(("co2_high", 2))
        elif v["co2"] >= T["co2_yellow"]:
            r.append(("co2_elevated", 1))
        if anomaly and not r:
            r.append(("unusual", 1))
        return sorted(r, key=lambda x: -x[1])

    # ---- main entry
    def update(self, reading, form=(7, 4, 2)):
        """form = (hours of sleep, mood 1-5 (5 good), stress 1-5 (5 high))"""
        self.win.append(reading.as_dict())
        v = pd.DataFrame(self.win).median().to_dict()      # smoothed vitals
        exertion = v["motion"] >= T["exertion_motion"]
        anomaly = (not exertion) and self.model.predict(pd.DataFrame([{k: v[k] for k in FEATURES}]))[0] == -1

        # physical
        raw = self._phys_raw(v, exertion, anomaly)
        raw_level = max([l for _, l in raw], default=0)
        phys_level = self.dp.step(raw_level, immediate=any(c in IMMEDIATE for c, _ in raw))
        reasons = [c for c, _ in raw] if phys_level > 0 else []
        if phys_level > 0 and not reasons:
            reasons = ["recovering"]

        # mental
        sleep, mood, stress_f = form
        s = 0.0 if exertion else self.stress_score(v)
        m_raw, m_reasons = 0, []
        if s >= 2.5:
            m_raw = 2
            m_reasons.append("sensor_high")
            if v["hr"] > 110:
                m_reasons.insert(0, "panic_pattern")
        elif s >= 1.0:
            m_raw = 1
            m_reasons.append("sensor_med")
        if (sleep < 6) + (mood <= 2) + (stress_f >= 4) >= 2:
            m_raw = min(2, m_raw + 1)
            m_reasons.append("form_bad")
        ment_level = self.dm.step(m_raw)
        if ment_level == 0:
            m_reasons = []
        elif not m_reasons:
            m_reasons = ["sensor_med"]

        return Assessment(phys_level, ment_level, reasons, m_reasons, round(s, 1), exertion, bool(anomaly),
                          {k: round(float(x), 1) for k, x in v.items() if k != "ts"})


def earth_packet(a: Assessment):
    """Prototype downlink report. Real system: encrypt (e.g. AES-GCM) before sending."""
    return {"type": "AETHER_RED_ALERT", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "physical": a.phys_reasons, "mental": a.ment_reasons, "vitals": a.vitals, "encrypted": False}


# ======================= CHATBOT =======================
MODEL_NAME = os.getenv("AETHER_MODEL", "qwen2.5:3b")  # alternatives: llama3.2:3b, gemma2:2b
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ----------------------------------------------------------------- language helpers
def detect_lang(text):
    return "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"


def norm(text):
    t = re.sub(r"[\u064B-\u0652\u0640]", "", text.lower())      # diacritics, tatweel
    t = re.sub(r"[أإآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return t


# ----------------------------------------------------------------- safety layer (fixed rules, not the LLM)
CRISIS = re.compile(
    r"(kill myself|suicid|end my life|hurt myself|self[- ]?harm|want to die|wanna die|don'?t want to live|better off dead"
    r"|انتحر|اقتل نفسي|(ا?ؤذي|ااذي|اذي|ايذاء)\s*نفسي|مش عايز اعيش|ما?\s*عايز اعيش|عايز اموت|نفسي اموت|"
    r"اخلص من حياتي|انهي حياتي|(?<!\w)اموت(?!\s*(من\s*)?(الضحك|الجوع|التعب|الحر|البرد|الخوف)))"
)
MEDICAL = re.compile(
    r"(chest pain|can'?t breathe|cannot breathe|difficulty breathing|short of breath|about to faint|passed out|numb|"
    r"ألم في صدري|الم في صدري|وجع في صدري|مش قادر اتنفس|مش عارف اتنفس|هغمي|هقع|دوخه شديده|تنميل)"
)
CRISIS_REPLY = {
    "en": "I'm really glad you told me, and I'm taking it seriously. I'm alerting your crew now so someone is with you. "
          "Please stay where you are and don't be alone. Can you tell me where you are right now?",
    "ar": "أنا ممتن أنك أخبرتني، وآخذ كلامك بجدية. أنبّه طاقمك الآن ليكون معك أحد. "
          "ابقَ في مكانك ولا تبقَ وحدك. هل تخبرني أين أنت الآن؟",
}
MEDICAL_REPLY = {
    "en": "That sounds physical, and it should not wait for a chat. Stop what you are doing, sit down, and tell a crewmate now. "
          "I'm flagging it on the monitor. Are you able to speak to someone right now?",
    "ar": "هذا يبدو عرضا جسديا ولا ينبغي انتظاره في محادثة. توقف عما تفعله، اجلس، وأبلغ زميلك الآن. "
          "سأضع تنبيها على شاشة المراقبة. هل تستطيع التحدث مع أحد الآن؟",
}

# ----------------------------------------------------------------- topics, protocols, fallbacks
TOPICS = {
    "sleep": r"sleep|insomnia|awake|exhaust|tired|nap|نوم|انام|بنام|نمت|ارق|منمتش|نعسان|سهران|ارهاق|مرهق",
    "anxiety": r"anxi|panic|nervous|worr|scared|afraid|stress|overwhelm|قلق|خايف|توتر|متوتر|هلع|مضغوط|ضغط|خوف",
    "low_mood": r"sad|down\b|depress|empty|hopeless|numb inside|حزين|زهقان|مكتئب|مخنوق|فاضي|يائس",
    "lonely": r"lonel|homesick|miss (my|home)|family|isolat|alone|وحيد|وحده|وحشتني|اهلي|بيتي|معزول",
    "conflict": r"argu|conflict|crewmate|annoy|angry|irritat|fight|زميل|خناق|مضايق|عصبي|متعصب|زعلان من",
    "focus": r"focus|concentrat|forget|foggy|distract|mistake|تركيز|انسي|ناسي|مشتت|بغلط",
}
PROTOCOL = {  # evidence-based, non-medical techniques the LLM may draw on (it adapts the wording)
    "sleep": "Sleep: keep a fixed sleep window, dim blue light 60 min before sleep, cool dark quarters, avoid caffeine late, "
             "short 20-min nap max if needed. If poor sleep lasts several days, tell the flight surgeon.",
    "anxiety": "Anxiety: paced breathing (in 4s, out 6s for 2 min); 5-4-3-2-1 grounding (name 5 things you see, 4 hear, 3 touch); "
               "label the feeling out loud; break the task into the next small step.",
    "low_mood": "Low mood: behavioural activation (one small meaningful task), a short exercise session, "
                "a call or message with someone you trust, sunlight-like cabin lighting. Persistent low mood: tell the flight surgeon.",
    "lonely": "Isolation: schedule a regular call with family, shared meals with crew, share a small task with a crewmate, "
              "keep a daily journal of good moments.",
    "conflict": "Conflict: pause before replying, name the specific behaviour not the person, use 'I' statements, "
                "take a short break, ask the commander or ground support to mediate if it repeats.",
    "focus": "Focus: single task for 25 minutes then 5 minute break, write the next step down, reduce interruptions, "
             "check sleep and hydration. Double-check critical procedures with a crewmate.",
    "general": "General: listen, reflect, ask what would help most right now.",
}
FALLBACK = {
    "sleep": {
        "en": "Poor sleep makes everything feel heavier, so that is a fair thing to feel. Tonight, try dimming the blue light an hour before bed and keeping a fixed sleep time. "
              "How many hours did you get last night?",
        "ar": "قلة النوم تجعل كل شيء أثقل، فشعورك مفهوم. الليلة جرّب تخفيف الإضاءة الزرقاء قبل النوم بساعة والتزم بموعد ثابت. "
              "كم ساعة نمت الليلة الماضية؟"},
    "anxiety": {
        "en": "That sounds uncomfortable, and it makes sense to want it to ease. Let's try two minutes of slow breathing together: in for 4, out for 6. "
              "What is the main thing your mind keeps returning to?",
        "ar": "يبدو ذلك مزعجا، ومن الطبيعي أن ترغب في أن يهدأ. لنجرّب دقيقتين من التنفس البطيء معا: شهيق 4 وزفير 6. "
              "ما أكثر شيء يشغل بالك الآن؟"},
    "low_mood": {
        "en": "I'm sorry it feels heavy right now. One small thing can help: pick one short, doable task or a quick walk through the module. "
              "What has been the hardest part of today?",
        "ar": "يؤسفني أن الأمر ثقيل عليك الآن. شيء صغير قد يفيد: اختر مهمة قصيرة سهلة أو تمشَّ قليلا داخل الوحدة. "
              "ما أصعب جزء في يومك؟"},
    "lonely": {
        "en": "Being far from home and people you love is a real strain, and missing them is natural. A scheduled call or a shared meal with the crew can help. "
              "Who would you most like to hear from?",
        "ar": "البعد عن البيت ومن تحب عبء حقيقي، والاشتياق إليهم طبيعي. موعد ثابت لمكالمة أو وجبة مشتركة مع الطاقم قد يساعد. "
              "من أكثر شخص تود سماع صوته؟"},
    "conflict": {
        "en": "Tension with a crewmate in a small space wears on anyone. Take a short pause before you respond, then describe the specific behaviour rather than the person. "
              "What happened?",
        "ar": "التوتر مع زميل في مساحة صغيرة يرهق أي شخص. خذ وقفة قصيرة قبل الرد، ثم صِف التصرف نفسه لا الشخص. "
              "ماذا حدث؟"},
    "focus": {
        "en": "Trouble concentrating is often a sign of fatigue or stress, not a lack of ability. Try 25 minutes on one task, then a 5-minute break, and double-check critical steps with a crewmate. "
              "How were sleep and hydration today?",
        "ar": "صعوبة التركيز غالبا علامة إرهاق أو توتر وليست ضعفا في قدرتك. جرّب 25 دقيقة على مهمة واحدة ثم 5 دقائق راحة، وراجع الخطوات الحرجة مع زميلك. "
              "كيف كان نومك وشربك للماء اليوم؟"},
    "general": {
        "en": "I'm here and listening. Tell me a bit more about what is going on, and we'll work out one small step together.",
        "ar": "أنا هنا وأسمعك. أخبرني أكثر عمّا يحدث، وسنجد معا خطوة صغيرة مناسبة."},
}

SYSTEM = (
    "You are AETHER, a private, on-device first-line support companion for an astronaut on a long mission. "
    "You listen, reflect and offer ONE practical step. You are not a doctor or therapist: never diagnose, never give "
    "medication or dosage advice. Tone: professional, warm, calm. Length: 2-4 sentences, under 60 words, plain language, "
    "no lists, no emojis. Reply in the same language the astronaut writes in (Arabic -> simple clear Arabic). "
    "Structure: (1) acknowledge what THEY specifically said, (2) one concrete technique from PROTOCOL if relevant, "
    "(3) one short follow-up question. Use CONTEXT only if it helps and never recite raw numbers unless asked. "
    "If they describe physical danger, tell them to alert a crewmate. If unsure, ask a clarifying question instead of guessing. "
    "Never invent mission facts."
)
SHOTS = {
    "en": [("I can't focus today and I keep making small mistakes.",
            "That sounds frustrating, especially when accuracy matters. Often it is fatigue rather than ability. "
            "Try 25 minutes on one task, then a 5-minute break, and double-check critical steps with a crewmate. How did you sleep?")],
    "ar": [("مش قادر أركز النهارده وبغلط في حاجات صغيرة.",
            "هذا مزعج، خصوصا عندما تكون الدقة مهمة. غالبا السبب إرهاق وليس ضعفا في قدرتك. "
            "جرّب 25 دقيقة على مهمة واحدة ثم 5 دقائق راحة، وراجع الخطوات الحرجة مع زميلك. كيف كان نومك؟")],
}
BAD_OUTPUT = re.compile(r"\b(mg|milligram|ibuprofen|paracetamol|acetaminophen|diazepam|xanax|sleeping pill|dose)\b|مسكن|حبوب|مللي|جرعه|جرعة")


# ----------------------------------------------------------------- logic

def classify(text):
    t = norm(text)
    scores = {k: len(re.findall(norm(p), t)) for k, p in TOPICS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def context_line(ctx):
    """ctx: dict(stress, ment_level, phys_level, sleep, mood, exertion)"""
    if not ctx:
        return ""
    lvl = ["normal", "elevated", "high"]
    return (f"CONTEXT: stress_indicator={ctx.get('stress', 0):.1f}/4 ({lvl[ctx.get('ment_level', 0)]}); "
            f"physical_status={lvl[ctx.get('phys_level', 0)]}; sleep_last_night={ctx.get('sleep', '?')}h; "
            f"mood={ctx.get('mood', '?')}/5; currently_exerting={'yes' if ctx.get('exertion') else 'no'}.")


def _valid(reply, lang):
    if not reply or len(reply) > 700 or BAD_OUTPUT.search(reply.lower()):
        return False
    return detect_lang(reply) == lang


def _llm(history, lang, topic, ctx):
    system = f"{SYSTEM}\n{context_line(ctx)}\nPROTOCOL: {PROTOCOL[topic]}"
    shots = []
    for u, a in SHOTS[lang]:
        shots += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    if os.getenv("ANTHROPIC_API_KEY"):  # demo option: stronger replies via API (not offline)
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=os.getenv("AETHER_API_MODEL", "claude-sonnet-5-5"), max_tokens=200, temperature=0.4,
            system=system, messages=shots + history[-8:])
        return r.content[0].text.strip()
    import ollama
    msgs = [{"role": "system", "content": system}] + shots + history[-8:]
    client = ollama.Client(host=OLLAMA_HOST, timeout=60)
    r = client.chat(model=MODEL_NAME, messages=msgs, keep_alive="30m",
                    options={"temperature": 0.4, "top_p": 0.9, "num_predict": 160, "num_ctx": 2048})
    return r["message"]["content"].strip()


def chat_reply(history, ctx=None):
    """history: [{'role','content'}...] with the latest user message last.
    returns (text, flags) where flags = {'crisis','medical','source','topic'}"""
    user = history[-1]["content"]
    lang = detect_lang(user)
    t = norm(user)
    if CRISIS.search(t):
        return CRISIS_REPLY[lang], {"crisis": True, "source": "rule"}
    if MEDICAL.search(t):
        return MEDICAL_REPLY[lang], {"medical": True, "source": "rule"}
    topic = classify(user)
    try:
        text = _llm(history, lang, topic, ctx)
        if _valid(text, lang):
            return text, {"source": "llm", "topic": topic}
    except Exception:
        pass
    return FALLBACK[topic][lang], {"source": "fallback", "topic": topic}


# ======================= APP (UI) =======================

st.set_page_config(page_title="Astronaut AI Doctor", layout="wide")
COLORS = ["#1e8e3e", "#f9ab00", "#d93025"]
LABELS = ["GREEN: all good", "YELLOW: attention", "RED: danger"]


def speak_async(text, lang):
    """Speaks in a separate process so the screen never freezes."""
    old = st.session_state.get("speech_proc")
    if old is not None and old.poll() is None:
        old.terminate()
    try:
        if shutil.which("espeak-ng"):
            cmd = ["espeak-ng", "-v", "ar" if lang == "ar" else "en-us", "-s", "145", text]
        else:
            code = "import pyttsx3,sys; e=pyttsx3.init(); e.setProperty('rate',150); e.say(sys.argv[1]); e.runAndWait()"
            cmd = [sys.executable, "-c", code, text]
        st.session_state.speech_proc = subprocess.Popen(cmd)
    except Exception as ex:
        st.session_state.speech_error = str(ex)


@st.cache_resource
def get_monitor():
    return Monitor(make_baseline())


monitor = get_monitor()

# ---------------- sidebar: simulated sensors (numbers only)
st.sidebar.title("Simulation")
voice_on = st.sidebar.checkbox("Voice alerts (speak automatically)", value=True)
lang = "ar" if st.sidebar.radio("Alert language", ["English", "العربية"], horizontal=True) == "العربية" else "en"
preset = st.sidebar.selectbox("Scenario", list(SCENARIOS))
p = SCENARIOS[preset]

with st.sidebar.expander("Sensor readings", expanded=False):
    k = preset
    v = {
        "hr": st.slider("Heart rate", 30, 200, int(p["hr"]), key=k + "hr"),
        "hrv": st.slider("HRV", 5, 80, int(p["hrv"]), key=k + "hrv"),
        "temp": st.slider("Temperature", 35.0, 40.0, float(p["temp"]), 0.1, key=k + "t"),
        "eda": st.slider("Sweat (EDA)", 0.5, 12.0, float(p["eda"]), 0.1, key=k + "e"),
        "resp": st.slider("Breathing", 8, 40, int(p["resp"]), key=k + "r"),
        "spo2": st.slider("Blood oxygen SpO2 (%)", 80, 100, int(p["spo2"]), key=k + "o"),
        "co2": st.slider("CO2 (ppm)", 400, 8000, int(p["co2"]), 50, key=k + "c"),
        "motion": st.slider("Motion (0 = still, 10 = heavy effort)", 0, 10, int(p["motion"]), key=k + "m"),
    }

with st.sidebar.expander("Daily check-in", expanded=False):
    form = (st.slider("Hours of sleep", 2, 10, 7), st.slider("Mood (5 = good)", 1, 5, 4),
            st.slider("Stress (5 = high)", 1, 5, 2))

a = monitor.update(Reading(**v), form)

# ---------------- page
st.title("Astronaut AI Doctor")
st.caption("Prototype on simulated data. Supports decisions, does not diagnose.")
tab1, tab2 = st.tabs(["Monitor", "Support chat"])


def card(title, level):
    st.markdown(
        f'<div style="background:{COLORS[level]};color:white;padding:16px;border-radius:12px;text-align:center">'
        f'<div style="font-size:14px">{title}</div><div style="font-size:24px;font-weight:600">{LABELS[level]}</div></div>',
        unsafe_allow_html=True)


with tab1:
    c1, c2 = st.columns(2)
    with c1:
        card("Physical health", a.phys_level)
    with c2:
        card("Mental health (indicators)", a.ment_level)
    st.write("")
    m = st.columns(7)
    m[0].metric("Heart rate", v["hr"])
    m[1].metric("Temp", v["temp"])
    m[2].metric("Breathing", v["resp"])
    m[3].metric("SpO2", f'{v["spo2"]}%')
    m[4].metric("CO2 ppm", v["co2"])
    m[5].metric("Motion", v["motion"])
    m[6].metric("Stress", "-" if a.exertion else f"{a.stress:.1f}/4")
    if a.exertion:
        st.caption("Heavy motion detected, so thresholds adapt and stress is not judged right now.")

    text = a.spoken(lang)
    if text:
        key = (a.phys_level, a.ment_level, tuple(a.phys_reasons), tuple(a.ment_reasons), lang)
        if voice_on and key != st.session_state.get("last_key"):  # speak only when the situation changes
            speak_async(text, lang)
            st.session_state.last_key = key
            st.toast("Speaking now...", icon="🔊")
        st.warning("🔊 SPEAKING: " + text)
        if "no_motion" in a.phys_reasons:
            st.error("CREW ALERT: crewmate is still and not responding. Go now and check consciousness and breathing.")
            st.caption("Report sent to Earth in parallel.")
        elif a.phys_level == 2:
            st.error("CREW ALERT: crewmate needs help immediately.")
            st.caption("Report sent to Earth in parallel.")
        elif a.ment_level >= 1:
            st.warning("Crew: your crewmate is under stress, keep an eye on them.")
        if a.ment_level >= 1:
            st.success("Want to talk? Open the 'Support chat' tab.")
        if st.button("Repeat voice alert"):
            speak_async(text, lang)
        if st.session_state.get("speech_error"):
            st.caption("Voice not available: " + st.session_state.speech_error)
    else:
        st.success("All readings are normal.")
        st.session_state.last_key = None

with tab2:
    st.write("A private space to talk. Arabic and English.")
    st.session_state.setdefault("chat", [])
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_text = st.chat_input("Type here...")
    if user_text:
        st.session_state.chat.append({"role": "user", "content": user_text})
        ctx = {"stress": a.stress, "ment_level": a.ment_level, "phys_level": a.phys_level,
               "exertion": a.exertion, "sleep": form[0], "mood": form[1]}
        with st.spinner("Thinking..."):
            answer, flags = chat_reply(st.session_state.chat, ctx)
        if flags.get("crisis") or flags.get("medical"):
            st.session_state.alert = "crisis" if flags.get("crisis") else "medical"
            if voice_on:
                ar = detect_lang(user_text) == "ar"
                speak_async("تنبيه للطاقم. زميلكم يحتاج مساعدة شخصية فورية." if ar
                            else "Alert to the crew. Your crewmate needs immediate personal support.", "ar" if ar else "en")
        st.session_state.chat.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.get("alert"):
        st.error("CREW ALERT sent: " + ("crewmate needs immediate personal support."
                                         if st.session_state.alert == "crisis" else "crewmate reports a physical symptom."))
        if st.button("Clear alert"):
            st.session_state.alert = None
            st.rerun()
