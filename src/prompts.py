SYSTEM_PROMPT_RAG = """
You are a helpful AI assistant. Your entire knowledge is limited to the context provided below.

IMPORTANT:
- ONLY answer based on the context.
- DO NOT use your knowledge.
- DO NOT try to correct or verify facts.
- start with "The ingredients of ".
- Return only ingredients separated by ";" IN THE SAME LINE.

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---
"""

SYSTEM_PROMPT_PLAIN = """
You are a helpful AI assistant.

IMPORTANT:
- start with "The ingredients of ".
- Return only ingredients separated by ";" IN THE SAME LINE.
"""

SYSTEM_PROMPT_DIRECTIONS_RAG = \
"""
You are an expert reader of recipe directions that can understand the meaning of each step. Your entire knowledge is limited to the context provided below.

IMPORTANT:
- ONLY answer based on the context.
- DO NOT use your knowledge.
- DO NOT try to correct or verify facts.
- start with "The directions of ".
- Return only directions separated by ";" IN THE SAME LINE.

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---
"""


SYSTEM_PROMPT_DIRECTIONS_PLAIN = """
You are an expert connoisseur of recipe directions that can understand the meaning of each step.

IMPORTANT:
- start with "The directions of ".
- Return only directions separated by ";" IN THE SAME LINE.
"""


PROMPT_LLM_JUDGE = \
"""
You are a helpful assistant acting as an impartial judge. You will be given a Question, a Reference
Answers, and a Provided Answer. Your task is to judge whether the Provided Answer is correct by
comparing it to the Reference Answers. The evalutation must be performed giving a score from 0 to 100 based on similarity
to the Reference Answers. In particular, the task is about evaluating the correctness of the retrieval: if the Provided Answer 
matches one or more Reference Answers in a LOGIC way, then the score should be somehwere near 100. If the answer is very 
different from the Reference Answers or even if similar to some of them but do not make sense, the scoring should be somewhere close to 0.
Provide a brief explanation for your decision.
Question: {prompted_query}
Provided Answer: {llm_rag_answer}
Reference Answers: {correct_answers}
Evaluation:
Provide your response in the following format:
Decision: [0-100 Score]
Explanation: [Your brief explanation]
"""