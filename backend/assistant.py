"""
Assistant: remembers the recent transcript, spots questions, and builds the LLM prompt
(recent transcript + matching chunks from your linked folders).
"""

from collections import deque

QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "who", "which", "can you", "could you", "would you",
    "do you", "does", "did you", "is there", "are there", "is it", "should", "will you",
    "have you", "tell me", "explain", "walk me through", "describe",
)

SYSTEM_PROMPT = (
    "You are a live meeting assistant helping the user (speaker 'Me') answer questions asked in a meeting. "
    "You get the recent transcript and excerpts from the user's own files and code. "
    "Answer the question directly in 2-5 short bullet points the user can say out loud. "
    "The user's linked files ARE their project: answer questions about 'your project' from them, "
    "using the file list and excerpts, and name the files you used. "
    "If the excerpts don't cover it, answer from general knowledge and say so in one short line. "
    "The transcript comes from speech recognition and may contain mistakes; infer the intended words."
)


def looks_like_question(text):
    t = text.lower().strip()
    return t.endswith("?") or t.startswith(QUESTION_STARTS)


class Assistant:
    def __init__(self, knowledge):
        self.kb = knowledge
        self.lines = deque(maxlen=40)
        self.last_question = None

    def add_line(self, speaker, text):
        self.lines.append(f"{speaker}: {text}")
        if looks_like_question(text):
            self.last_question = text

    def build_prompt(self, question):
        recent = "\n".join(list(self.lines)[-12:])   # shorter prompt = faster first word
        # Search with the question itself; short questions ("why?") borrow the line before.
        query = question
        if len(question.split()) < 5 and len(self.lines) >= 2:
            query = self.lines[-2] + " " + question
        hits = self.kb.search(query, k=3)
        if hits:
            refs = "\n\n".join(f"[{i + 1}] {src}\n{txt}" for i, (src, txt) in enumerate(hits))
        else:
            refs = "(no matching excerpts)"
        files = self.kb.files[:50]
        file_map = "\n".join(files) + ("\n..." if len(self.kb.files) > 50 else "") if files else "(no folders linked)"
        user = (
            f"Files in my linked project folders:\n{file_map}\n\n"
            f"Most relevant excerpts from those files:\n{refs}\n\n"
            f"Recent meeting transcript:\n{recent}\n\n"
            f"Question to answer now: {question}"
        )
        return SYSTEM_PROMPT, user
