"""
app/utils/questionnaire_config.py
==================================
40-question behavioral questionnaire configuration.

Matches the frontend questionsBank exactly.
Questions Q1–Q20 (Social + Communication) are positive-behavior indicators
  → absence of behavior indicates risk → scores are INVERTED before model input.
Questions Q21–Q40 (Behavior + Sensory) are risk-behavior indicators
  → presence of behavior indicates risk → scores used as-is.

Scale: 0 = Never | 1 = Rarely | 2 = Sometimes | 3 = Often | 4 = Always
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionItem:
    """A single screening question."""

    id: int  # 1-based (Q1 … Q40)
    text: str
    section: str  # Human-readable section name
    section_index: int  # 0-based section index
    autism_indicator: str  # "low" = positive behaviour (inverted), "high" = risk behaviour


@dataclass(frozen=True)
class SectionInfo:
    """Metadata for a question section."""

    name: str
    index: int
    start_q: int  # 1-based inclusive
    end_q: int  # 1-based inclusive
    autism_indicator: str  # "low" or "high"


# ── Section definitions ─────────────────────────────────────────
SECTIONS: list[SectionInfo] = [
    SectionInfo("Social Interaction", 0, 1, 10, "low"),
    SectionInfo("Communication", 1, 11, 20, "low"),
    SectionInfo("Behavior Patterns", 2, 21, 30, "high"),
    SectionInfo("Sensory & Emotional", 3, 31, 40, "high"),
]

# ── Question bank (matches frontend questionsBank 1:1) ──────────
_RAW_QUESTIONS: list[list[str]] = [
    # Section 0 — Social Interaction (Q1–Q10)
    [
        "Does your child respond to their name?",
        "Does your child make eye contact?",
        "Does your child smile back when smiled at?",
        "Does your child show interest in other children?",
        "Does your child try to share enjoyment with you?",
        "Does your child point to show interest?",
        "Does your child wave goodbye?",
        "Does your child imitate actions?",
        "Does your child follow your gaze?",
        "Does your child bring objects to show you?",
    ],
    # Section 1 — Communication (Q11–Q20)
    [
        "Does your child use gestures to communicate?",
        "Does your child understand simple instructions?",
        "Does your child babble or speak words?",
        "Does your child respond when spoken to?",
        "Does your child use meaningful sounds?",
        "Does your child ask for help when needed?",
        "Does your child respond to emotions?",
        "Does your child engage in back-and-forth sounds?",
        "Does your child use facial expressions?",
        "Does your child try to start conversations?",
    ],
    # Section 2 — Behavior Patterns (Q21–Q30)
    [
        "Does your child repeat actions again and again?",
        "Does your child show unusual attachment to objects?",
        "Does your child line up toys?",
        "Does your child get upset with small changes?",
        "Does your child show repetitive movements (e.g. hand flapping)?",
        "Does your child focus on parts of objects?",
        "Does your child spin or rock frequently?",
        "Does your child insist on routines?",
        "Does your child play with toys in unusual ways?",
        "Does your child show intense interest in specific things?",
    ],
    # Section 3 — Sensory & Emotional (Q31–Q40)
    [
        "Does your child react strongly to loud sounds?",
        "Does your child avoid eye contact?",
        "Does your child show limited emotional expression?",
        "Does your child overreact to touch?",
        "Does your child ignore pain or temperature?",
        "Does your child get easily frustrated?",
        "Does your child have difficulty calming down?",
        "Does your child show fear without reason?",
        "Does your child avoid social interaction?",
        "Does your child prefer to play alone?",
    ],
]

# ── Build flattened question list ────────────────────────────────
QUESTIONS: list[QuestionItem] = []
for section in SECTIONS:
    for offset, text in enumerate(_RAW_QUESTIONS[section.index]):
        QUESTIONS.append(
            QuestionItem(
                id=section.start_q + offset,
                text=text,
                section=section.name,
                section_index=section.index,
                autism_indicator=section.autism_indicator,
            )
        )

assert len(QUESTIONS) == 40, f"Expected 40 questions, got {len(QUESTIONS)}"

# ── Indices that need score inversion (0-based) ─────────────────
# Q1–Q20 are positive-behaviour questions → invert scores
INVERT_INDICES: set[int] = set(range(0, 20))

# ── Scale labels (for API response) ─────────────────────────────
SCALE_LABELS: dict[int, str] = {
    0: "Never",
    1: "Rarely",
    2: "Sometimes",
    3: "Often",
    4: "Always",
}

TOTAL_QUESTIONS = 40
MIN_RESPONSE_VALUE = 0
MAX_RESPONSE_VALUE = 4
