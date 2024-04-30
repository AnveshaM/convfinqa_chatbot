from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_single_hop_questions():
    assert query_and_validate(
        question="what is the number of shares to be issued upon exercise of outstanding options warrants and rights is approved by security holders?",
        expected_response="2956907",
    )

def test_simple_add_questions():
    assert query_and_validate(
        question="what was, then, in millions, the total tax of benefit realized from option exercises under all incentive plans for both years, 2013 and 2014 combined?",
        expected_response="64",
    )

def test_simple_subtract_questions():
    assert query_and_validate(
        question="what is the net difference in gains between the price of advance auto and the price of the s&p index in 2012?",
        expected_response="62.63",
    )

def test_simple_multiply_questions():
    assert query_and_validate(
        question="what is the total fair value of warrants and rights that are issued and approved by by security holders?",
        expected_response="247470105.4",
    )

def test_simple_divide_questions():
    assert query_and_validate(
        question="what portion of total number of securities issued is approved by security holders?",
        expected_response="0.96",
    )
def test_multiple_hop_questions():
    assert query_and_validate(
        question="what is the portion of total number of facilities located in the united states?",
        expected_response="48.8%",
    )
def test_single_hop_questions_neg():
    assert query_and_validate(
        question="what is the number of shares to be issued upon exercise of outstanding options warrants and rights is approved by security holders?",
        expected_response="29569",
    )

def test_simple_add_questions_neg():
    assert query_and_validate(
        question="what was, then, in millions, the total tax of benefit realized from option exercises under all incentive plans for both years, 2013 and 2014 combined?",
        expected_response="90",
    )

def test_simple_subtract_questions_neg():
    assert query_and_validate(
        question="what is the net difference in gains between the price of advance auto and the price of the s&p index in 2012?",
        expected_response="55",
    )

def test_simple_multiply_questions_neg():
    assert query_and_validate(
        question="what is the total fair value of warrants and rights that are issued and approved by by security holders?",
        expected_response="9999999",
    )

def test_simple_divide_questions_neg():
    assert query_and_validate(
        question="what portion of total number of securities issued is approved by security holders?",
        expected_response="0.2",
    )
def test_multiple_hop_questions_neg():
    assert query_and_validate(
        question="what is the portion of total number of facilities located in the united states?",
        expected_response="98.8%",
    )



def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )