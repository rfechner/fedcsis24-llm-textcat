from typing import Any, List
import ollama
from abc import abstractmethod
from src.util.llm_utils import extract_json_from_string
from collections import Counter

class PromptManager:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run_prompt(self, messages, model_id):
        raise NotImplementedError

class IdentityPrompt(PromptManager):
    def __init__(self) -> None:
        pass

    def run_prompt(self, messages, model_id):
        return ollama.chat(
                model=model_id,
                messages=messages
            )

class VerifyValidClass(PromptManager):
    """
        Implicitly also verfies that the output is json.
    """
    def __init__(self, extract_ypred_fn, valid_classes : List[str]) -> None:
        super().__init__()

        self.valid_classes = valid_classes

        # raises ValueError if retrieved class not in valid_classes
        self.condition_function = lambda x: \
            self.valid_classes.index(extract_ypred_fn(
                extract_json_from_string(x), throw=True, ret_label=True,
            ))

    def run_prompt(self, messages, model_id, patience=3):
        
        while patience >= 0:
            outputs = ollama.chat(
                model=model_id,
                messages=messages
            )
            assistant_response = outputs['message']['content']
            try:
                _ = self.condition_function(assistant_response)
                return outputs
            
            # Output illegal. Could throw KeyError or ValueError.
            except Exception:
                patience -= 1
        raise ValueError(f"Trying to re-run prompts for message {messages} failed. Output not in valid classes.")

class VerifyJsonOutput(PromptManager):

    def __init__(self) -> None:
        super().__init__()
        self.condition_function = extract_json_from_string

    def run_prompt(self, messages, model_id, patience=3):
        
        while patience >= 0:
            outputs = ollama.chat(
                model=model_id,
                messages=messages
            )
            assistant_response = outputs['message']['content']
            try:
                # TODO: in ecmv: why is assistant_response == ''?
                _ = self.condition_function(assistant_response)
                return outputs
            
            # couldn't parse json. Repeat chat.
            except ValueError:
                patience -= 1
        raise ValueError(f"Trying to re-run prompts for message {messages} failed. Couldn't parse json.")

class GatherHypotheses(PromptManager):

    def __init__(self, n : int = 5, verify_json : bool = False) -> None:
        super().__init__()
        self.n = n
        self.verify = VerifyJsonOutput() if verify_json else IdentityPrompt()

    def run_prompt(self, messages, model_id):
        
        votes = []
        outputs = []

        # Accumulate n runs.
        for _ in range(self.n):    
            try:
                output = self.verify.run_prompt(messages, model_id)
                assistant_response = output['message']['content']
                json_dict = extract_json_from_string(assistant_response)

            # couldn't parse json
            except ValueError:
                continue
        
            votes.append(json_dict)
            outputs.append(output)

        if len(votes) == 0:
            raise ValueError("Couldn't extract single hypothesis.")
        
        # gather a list of json responses
        hypotheses = '\n'.join([str(v) for v in votes])
        response = f"Based on the given information, these are some hypotheses:\n{hypotheses}"
        
        output = outputs[-1].copy()
        output['message']['content'] = response

        return output
            
class SequentialVerify(PromptManager):

    def __init__(self, *pms) -> None:
        super().__init__()
        self.verification_classes = [pm.__class__ for pm in list(*pms)]
        self.conditions = [pm.condition_function for pm in list(*pms)]

    def run_prompt(self, messages, model_id, patience = 5):

        while patience >= 0:
            outputs = ollama.chat(
                model=model_id,
                messages=messages
            )
            assistant_response = outputs['message']['content']
            try:
                # iterate over condition functions, if one fails, the whole verification fails.
                for fn in self.conditions:
                    _ = fn(assistant_response)

                return outputs
            except ValueError:
                patience -= 1
        raise ValueError(f"Trying to re-run prompts for message {messages} failed. Multiple verifications: {self.verification_classes} failed.")
    
    