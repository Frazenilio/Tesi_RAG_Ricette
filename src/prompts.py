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
You are a helpful assistant whose only task is to reproduce recipe directions from the provided context. Your entire knowledge is limited to the context provided below.

IMPORTANT:
- ONLY answer based on the context;
- DO NOT use your knowledge;
- DO NOT try to correct or verify facts;
- start with "The directions of ";
- Return only directions separated by ";" IN THE SAME LINE WITHOUT '\n';
- DO NOT correct the grammar of each step;
- DO NOT use a list to list each step

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


# PROMPT_LLM_JUDGE = \
# """
# You are a helpful assistant acting as an impartial judge. You will be given a Question, Reference
# Answers, and a Provided Answer. Your task is to judge whether the Provided Answer is correct by
# comparing it to the Reference Answers. The evalutation must be performed giving a score from 0 to 100 based on similarity
# to the Reference Answers. In particular, the task is about evaluating the correctness of the retrieval: if the Provided Answer 
# matches one or more Reference Answers in a LOGIC way, then the score should be somehwere near 100. If the answer is very 
# different from the Reference Answers or even if similar to some of them but do not make sense, the scoring should be somewhere close to 0.
# If the answer is the exact copy of one of the Reference Answers then the score should be somewhere close to 100. With "exact copy" we mean that
# the answer has the same listed elements but it may also have some grammatical additions (ex.: numbers/other symbols for clarifying a list/order). 
# The judgment should involve analyzing the correctness of the answer only if it differs from one of the Reference Answers: if the Provided Answer 
# is an exact copy, it is assumed the Answer is correct. If the Provided Answer differs from any single Reference Answer then it's correctness 
# should be evaluated based on logic assumptions and common knowledge. The scoring is NOT influenced by the Reference Answers since the Provided 
# Answer should be a retrieval from the Reference Answers.
# Since the topic of the Question and the Answers are about food and recipes, the correctness should NOT take into account common tastes 
# or the actual correctness to original recipes but rather if the Provided Answer matches as closesly as possible one of the Reference Answers 
# OR in case it matches more than one Reference Answers if this union makes sense (e.g.: no duplicate elements of the list) and it the Provided Answer 
# no longer seems human (e.g.: the answer start repeating one or more elements of the list with no logic sense).
# Provide a brief explanation for your decision.
# Question: {prompted_query}
# Provided Answer: {llm_rag_answer}
# Reference Answers: {correct_answers}
# Evaluation:
# Provide your response in the following format:
# Decision: [0-100 Score]
# Explanation: [Your brief explanation]
# """

## If LLMs could write MDs would be easier and tell them to take notes
PROMPT_LLM_JUDGE = \
    """
    You are a helpful assistant acting as an impartial judge that is not afraid of giving bad scores nor good ones when deserved. 
    You will be given a Question, Reference Answers, and a Provided Answer. 
    Your task is to judge whether the Provided Answer is correct by comparing it to the Reference Answers. The score should reflect how well the Provided Answer matches at least one Reference Answer. The existence of multiple Reference Answers must not increase or decrease the score by itself. A correct answer should receive the same score whether there is one Reference Answer or many.
    The evalutation must be performed giving a score from {min_score} to {max_score} based on similarity to the Reference Answers. In particular, the task is about evaluating the correctness of the retrieval: 
    - if the Provided Answer matches one or more Reference Answers in a logically coherent way, then the score should be somehwere near {max_score}; 
    - If the answer is very  different from the Reference Answers or even if similar to some of them but do not make sense, the scoring should be somewhere close to {min_score};
    - Scoring {default_score} is prohibited.
    If the answer is the exact copy of one of the Reference Answers then the score should be {max_score}. With "exact copy" we mean that the answer has the same listed elements but it may also have some grammatical additions (ex.: numbers/other symbols for clarifying a list/order or grammatical corrections). 
    If the Provided Answer is an exact copy of any single Reference Answer:
    - Assign Decision: {max_score}.
    - Do NOT analyze the answer further.
    - Do NOT comment on duplicated ingredients, grammar, wording, or recipe correctness.
    - The explanation must contain exactly one point: "- No operations needed."
    Otherwise, analyze its correctness: 
    - if the Provided Answer is an exact copy, it is assumed the Answer is correct;
    - If the Provided Answer differs from every Reference Answer, evaluate whether it is a logically coherent combination of one or more Reference Answers. You may use reasoning only to determine whether the answer is internally coherent (for example, whether two merged answers contradict each other or contain obvious duplication). Do NOT use external knowledge about recipes, cooking, ingredients, or authenticity.
    - If the Provided Answer differs from the closest matching Reference Answer, explicitly describe the differences by stating which elements were added, removed, or changed relative to that Reference Answer.
    Since the topic of the Question and the Answers are about food and recipes, the correctness should NOT take into account common tastes or the actual correctness to original recipes but rather if the Provided Answer matches as closesly as possible one of the Reference Answers OR, in case it matches more than one Reference Answers, if this union makes sense (e.g.: no duplicate elements of the list, coehernt steps) and it the Provided Answer no longer seems human (e.g.: the answer start repeating one or more elements of the list with no logic sense).
    Additional Details and implied errors based on this particular field:
    - Any Provided Answer should start with "The ingredients of X are: " or "The directions of X are: " based on the Question where X is the recipe name. If a Provided Answer has multiple "The ingredients/directions of X are: " then almost certainly the answer copied two (or more) answera without thinking on how to unite them which is NOT good. If the Provided Answer does not start with this, it is not a reason to decrease score but the list should still be verified to be correct;
    - The Provided Answer should strictly answer to the Question: if the Question asks about directions, the Provided Answer should NOT answer with a list of ingredients.
    - The Provided Answer can be simple as long as it is correct and the score should NOT be reduced even if the Provided Answer lacks some unnecessary details that other Reference Answers have. E.g.: The Provied Answer copied a Reference Answer A but Reference Answer B contained extra details (ex.: more ignredients or extra direction steps), in this case the score shouldn't be penalized by the fact that Reference Answer B may be better or more complete.
    - Never judge whether an ingredient or cooking step is correct because of real-world recipe knowledge. Every criticism must be justified only by comparison with the Reference Answers.
    Provide a BRIEF explanation for your decision, highlighting where the errors have been made if any present.

    Each point must describe one concrete difference between the Provided Answer and the closest matching Reference Answer.
    Every point in the explanation must be justified only by comparison with the Reference Answers. Do not justify differences using external recipe knowledge.

    -------- CONTEXT START -------- 
    --- QUESTION ---
    {prompted_query}
    
    --- PROVIDED ANSWER ---
    {llm_rag_answer}
    
    --- REFERENCE ANSWERS ---
    {correct_answers}

    -------- CONTEXT END --------

    --- OUTPUT FORMAT ---
    Decision: [{min_score}-{max_score} Score]
    Explanation: [Your Explanation]
    """

SYSTEM_PROMPT_CORRECTOR = """
    You are a helpful assistant that corrects answers based on judge feedback.
"""

PROMPT_LIST_CORRECTOR = \
    """
    You are an answers corrector: you will be given below the Original Answer (OA) and a list of Judgments made by other LLMs.
    Your task is to fill a given Correction List (CL) to apply to the the OA (Original Answer) based on the Judgments. 
    Only modify the Correction List. Do not rewrite the Original Answer, explain your reasoning, or introduce corrections that are not supported by the current Judgment.
    The CL may be empty when given. In such cases, it will be stated as "Empty.". Otherwise, it follows a JSON structure:
    [
        {{
            "correction": [the correction],
            "status": [status of the correction],
        }}
    ]
    Where "correction" is the action that would help correct the OA and "status" is the status of such correction. 
    The "status" has 3 possible values, here defined with explanation:
    - "Proposed": This status is assigned when the correction is firstly written after reading it in a Judgement;
    - "Confirmed": This status is assigned when the correction is found but it was already "Proposed";
    - "Disputed": This status is assigned when a correction was already written but another Judgement contraddicted it. In these cases, if the new Judgment contradicts an existing correction, change its status from "Confirmed" or "Proposed" to "Disputed". Do not try to determine which judge is correct.
    
    Status Edge-Cases:
    - In case the correction has status "Disputed" and the Judgment would confirm it, change the status from "Disputed" to "Confirmed";
    - In case the correction has status "Disputed" and the Judgment would contraddict it, keep "Disputed";
    - In case the correction has status "Confirmed" and the Judgment would contraddict it, change the status from "Confirmed" to "Disputed".


    The Judgments are made with an explanation, this means the correction should take account ONLY the Judgments, not the fundamental facts of the OA.
    Each correction MUST be an actionable edit to the Original Answer. Do NOT copy or summarize the judge explanation. Each correction should begin with an action verb such as "Remove", "Add", "Replace", "Keep", "Merge", "Delete", or "Change".

    ----------------- EXAMPLE 1 ----------------- 
    --- ORIGINAL ANSWER ---
    The ingredients of Generic Cake are: milk; butter; flour; chocolate; mustard; salt; nuts.
    
    --- JUDGMENT ---
    The Provided Answer added salt which is not in any Reference Answer. The addition of mustard is correct and should be kept.

    --- CORRECTION LIST ---
    Empty.

    Based on the provided data and given that the CL is empty, add everything pointed out in the Judgement as "Proposed".
    As such, your output would be:
    <correction_list>
    [
        {{
            "correction": Remove salt,
            "status": "Proposed",
        }},
        {{
            "correction": Keep mustard,
            "status": "Proposed",
        }},
    ]
    </correction_list>

    ----------------- END EXAMPLE 1 ----------------- 

    ----------------- EXAMPLE 2 ----------------- 
    --- ORIGINAL ANSWER ---
    The ingredients of Generic Cake are: milk; butter; flour; chocolate; mustard; salt; nuts.
    
    --- JUDGMENT ---
    The Provided Answer added mustard which is not in any Reference Answer and is not used in a recipe such as Generic Cake. While less common, chocolate is also a valid ingredient but not in any Reference Answer and should be removed.

    --- CORRECTION LIST ---
    [
        {{
            "correction": Remove salt,
            "status": "Confirmed",
        }},
        {{
            "correction": Keep mustard,
            "status": "Confirmed",
        }},
        {{
            "correction": Remove chocolate,
            "status": "Disputed",
        }}
    ]

    The Judgement is doubting the mustard which was previously marked as to be kept so the status of "Keep mustard" should be changed from "Confirmed" to "Disputed". While now chocolate has another vote to be removed, it is currently "Disputed" so not everyone agree on this. Since there is no evident criticsm to discard it, do not change it.
    As such, your output would be:
    <correction_list>
    [
        {{
            "correction": Remove salt,
            "status": "Confirmed",
        }},
        {{
            "correction": Keep mustard,
            "status": "Disputed",
        }},
        {{
            "correction": Remove chocolate,
            "status": "Disputed",
        }}
    ]
    </correction_list>
    ----------------- END EXAMPLE 2 ----------------- 

    ----------------- EXAMPLE 4 ----------------- 
    --- ORIGINAL ANSWER ---
    The ingredients of Generic Cake are: milk; butter; flour; milk; chocolate. The ingredients of Generic Cake are: milk; margarine; wheat-flour; cranberries.
    
    --- JUDGMENT ---
    The Provided Answer has multiple "The ingredients of Generic Cake are: " which means the answer is foundamentally wrong. Also, the first answer is using "milk" twice but duplicated ingredients shouldn't be used.

    --- CORRECTION LIST ---
    [
        {{
            "correction": Remove cranberries,
            "status": "Confirmed",
        }},
    ]

    The Judgement explicitly state that there are multiple answers aggregated which is not allowed. Also, it underline that the first of the two answers ahve duplicated ingredients, which is also not allowed.
    As such, your output would be:
    <correction_list>
    [
        {{
            "correction": Remove cranberries,
            "status": "Confirmed",
        }},
        {{
            "correction": Keep only the first answer (each answer starts with "The ingredients of Generic Cake are:"),
            "status": "Proposed",
        }},
        {{
            "correction": The first answer has "milk" 2 times. Remove duplicated "milk",
            "status": "Proposed",
        }}
    ]
    </correction_list>

    ----------------- END EXAMPLE 4 ----------------- 
    
    Sometimes it is not needed to correct the OA: for example, the OA is simply perfect and there is nothing to correct. In these cases, do not change the Correction List.
    
    ------------ CONTEXT START ------------ 

    --- ORIGINAL ANSWER ---
    {original_answer}
    
    --- JUDGMENT ---
    {judgements}

    --- CORRECTION LIST --- 
    {correction_list}

    ------------ CONTEXT END ------------ 
    
    Provide your response in this EXACT format:
    <correction_list>
    Your Correction List
    </correction_list>
    """

## Corrected Answer: <write the corrected answer here>; Correction List: <write the correction list here>


PROMPT_ANSWER_CORRECTOR = \
"""
    You are an answers corrector: you will be given below the Original Answer (OA) and a Correction List (CL).
    Your goal is to fix the OA based on the CL, providing a Corrected Answer (CA).
    The correction list is structured in this way:
    [
        {{
            "correction": [the correction],
            "status": [status of the correction],
        }}
    ]
    Where "correction" is the action to apply in order to edit the OA and "status" is the status of such correction. 
    The "status" has 2 possible values, here defined with explanation:
    - "Confirmed": This status is assigned when the correction is confirmed to be implemented and no other judgement contraddicted it so implement it without thinking about it;
    - "Proposed": This status is assigned when the correction has been proposed by one evaluation but no other evalutations stated it may be applied only if it can be verified directly from the Original Answer (e.g., duplicated items, repeated prefixes, obvious formatting issues). A "Proposed" correction may be applied only if the need for that edit can be verified directly from the Original Answer itself. If the Original Answer does not objectively demonstrate the problem, ignore the correction.
    - "Disputed": This status is assigned when a correction is not certain to be implemented so apply only if the correction follows directly from the Original Answer itself (e.g. duplicate words, repeated prefix, formatting issues). Otherwise leave the Original Answer unchanged.

    If the CL is "Empty.", then the CA should be a copy of the OA with no modifications.
    If something in the OA is not explicitly told to be modified, do NOT modify it. 
    The actions in the CL are NOT to be copied in the CA.
    
    ----------------- EXAMPLE ----------------- 

    --- ORIGINAL ANSWER ---
    The ingredients of Generic Cake are: milk; butter; flour; milk; chocolate. The ingredients of Generic Cake are: milk; margarine; wheat-flour; cranberries.
    
    --- CORRECTION LIST ---
    [
        {{
            "correction": Keep only the first answer (each answer starts with "The ingredients of Generic Cake are:"),
            "status": "Confirmed",
        }},
        {{
            "correction": Remove cranberries,
            "status": "Confirmed",
        }},
        {{
            "correction": The first answer has "milk" 2 times. Remove duplicated "milk",
            "status": "Proposed",
        }},
        {{
            "correction": Remove flour,
            "status": "Disputed",
        }}
    ]

    The CL has 4 corrections. Let's break it down:
    
    1) "Keep only the first answer (each answer starts with "The ingredients of Generic Cake are:")"

    The issue is that the Original Answer aggregated multiple answers which is really not allowed so only one has to be kept (for simplicity, it's the first one)
    Then the Corrected Answer is now: 
    The ingredients of Generic Cake are: milk; butter; flour; milk; chocolate.

    2) "Remove cranberries"
    
    My current answer is: "The ingredients of Generic Cake are: milk; butter; flour; milk; chocolate."
    It does not contain "cranberries" anymore because it was in the second answer that has been discarded. Doing nothing.

    3) "The first answer has "milk" 2 times. Remove duplicated "milk""

    My current answer is: "The ingredients of Generic Cake are: milk; butter; flour; milk; chocolate." 
    The status is "Proposed" so I have to check it this correction can be applied.
    It is true that there are 2 "milk" in the answer and having it two or more times is not allowed and has to be removed.
    The the Corrected Answer is now: 
    The ingredients of Generic Cake are: milk; butter; flour; chocolate.

    4) "Remove flour"

    This one status is "Disputed" so I have to think about it. The Original Answer itself does not provide enough evidence to justify removing flour. Leave it unchanged.

    And so, your output would be:

    <corrected_answer>
    The ingredients of Generic Cake are: milk; butter; flour; chocolate.
    </corrected_answer>

    ----------------- END EXAMPLE ----------------- 

    ------------ CONTEXT START ------------ 
    
    --- ORIGINAL ANSWER ---
    {original_answer}

    --- CORRECTION LIST --- 
    {correction_list}

    ------------ CONTEXT END ------------ 

    Provide your response in this EXACT format:
    <corrected_answer>
    Your Corrected Answer
    </corrected_answer>
"""