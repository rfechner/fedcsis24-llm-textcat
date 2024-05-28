import warnings
import copy

from typing import Tuple, List
from src.prompts.prompt_examples import *
from src.prompts.prompt_manager import (
    PromptManager, IdentityPrompt, VerifyJsonOutput, GatherHypotheses, VerifyValidClass
)

from abc import abstractmethod


class PromptStrategy:
    def __init__(self, ptype, id2label, label2id, valid_classes : List[str]):
        self.ptype = ptype
        self.id2label = id2label
        self.label2id = label2id
        self.valid_classes = valid_classes
        self.mappings = get_valid_mappings(self.valid_classes)

    @abstractmethod
    def get_messages(self, text : str,  label : str = None) -> Tuple[str, List[Tuple[str, PromptManager]]]:
        raise NotImplementedError

    @abstractmethod
    def extract_ypred_from_json(self, json : dict):
        raise NotImplementedError
    
    @staticmethod
    def create_from_string(ps: str, **kwds):
        __ps_mapper__ = {
            "direct": DirectStrategy,
            "ec" : ExtractClassifyStrategy,
            "ecr" : ExtractClassifyReflectStrategy,
            "ech" : ExtractClassifyHypotheses
        }

        assert ps in set(__ps_mapper__.keys())
        assert 'ptype' in kwds.keys() and kwds['ptype'] in ['zero_shot', 'one_shot', 'few_shot']

        return __ps_mapper__[ps](**kwds)

    def pre_prompt(self, instruction, answers, thoughts=None):
        if self.ptype == "zero_shot":
            return ''
        elif self.ptype == 'one_shot':
            return f"""
            Example:
            {instruction}
            {example1}
            {answers['example1']}\n"""
        elif self.ptype == 'few_shot':
            return f"""
            Example 1:
            {instruction}
            {example1}
            {answers['example1']}\n
            Example 2:
            {instruction}
            {example2}
            {answers['example2']}\n
            Example 3:
            {instruction}
            {example3}
            {answers['example3']}\n"""
        elif self.ptype == 'cot':
            raise NotImplementedError
            return f"""
            Example 1:
            {instruction}
            {example1}\n
            {thoughts['example1']}\n
            {answers['example1']}\n
            Example 2:
            {instruction}
            {example2}\n
            {thoughts['example2']}\n
            {answers['example2']}\n
            Example 3:
            {instruction}
            {example3}\n
            {thoughts['example3']}\n
            {answers['example3']}\n"""
        else:
            raise NotImplementedError(f'Not implemented for ptype {self.ptype}')
    
    def extract_ypred_from_json(self, json : dict, key='section_letter', throw=False, ret_label=False) -> int:
        try:
            if ret_label:
                return json[key]
            else:
                return self.label2id[json[key]]
        except KeyError:
            if throw:
                raise
            warnings.warn(f'Encountered invalid section_letter {json[key]}, returning default value of 0')
            return 0
        
class DirectStrategy(PromptStrategy):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.system_prompt = "You're a helpful expert on the WZ08, the taxonomy of the industrial sectors of Germany. \
            Your job is to classify the WZ08 industrial section of the company posting the jobad."
        self.setup = """Given is the following jobposting:\n{}"""
        self.instruction = \
        f"\nPlease classify the WZ08 industrial section of the company posting the jobad. Answer in json using keys 'industrial_section' (e.g. Vermittlung von Arbeitskräften), 'section_letter' (CHOOSE ONLY FROM: {self.valid_classes}) and 'reasoning'. Answer:"
        self.answers = {
            'example1' : 
            """
            Answer:
            {
                "industrial_section" : "Handel; Instandhaltung und Reperatur von Kraftfahrzeugen",
                "section_letter" : "G"
                "reasoning" : "The employer, hagebau Handelsgesellschaft für Baustoffe mbH & Co. KG is a leading company in the industrial section for 'Baustoff-, Holz- und Fliesenhandel' which matches the description of the industrial section G",
            }
            """,
            'example2' : 
            """
            Answer:
            {
                "industrial_section" : "Vermittlung und Überlassung von Arbeitskräften",
                "section_letter" : "N"
                "reasoning" : "The employer, ELSTA GmbH & Co. KG, is involved in the Vermittlung von Arbeitskräften (placement of workers) industry, which falls under the broader category of 'Vermittlung und Überlassung von Arbeitskräften'.",
            }
            """,
            'example3' : 
            """
            Answer:
            {
                "industrial_section" : "Baugewerbe",
                "section_letter" : "F"
                "reasoning" : "The employer, Koch Dachtechnik GmbH, is involved in 'Dachdeckerei und Bauspenglerei' which matches section F.",
            }
            """
        }
        self.thoughts = {
            'example1' : """
            Thoughts:
            
            hagebau Handelsgesellschaft für Baustoffe mbH & Co. KG is a leading cooperative in the building materials, wood, 
            flooring, and DIY sectors with over 1,500 locations across Germany, Austria, Switzerland, France, Luxembourg, Belgium, and
            Spain. The company has approximately 1,400 employees and offers various services through its subsidiaries, including 
            insurance, logistics, consulting, and more.
            """,
            'example2' : """
            Thoughts:
            
            The ELSTA GmbH & Co. KG company is looking for experienced roofers/ carpenters (m/w) to be hired immediately, with a focus
            on placements in southern Germany. The job involves assisting with tasks such as roofing, renovation, and maintenance 
            work, requiring physical fitness, teamwork, and independent working abilities.
            """,
            'example3' : """
            Thoughts:
            
            The Koch Dachtechnik GmbH is one of the leading companies for waterproofing, trapezoidal sheet and facade work, with 
            locations in Saxony, Saxony-Anhalt, Lower Saxony, Thuringia, and Rhineland-Palatinate. The company was founded in 1878 and
            has a medium-sized enterprise with a strong reputation in the industry.
            """
        }

    def get_messages(self, text : str,  label : str = None):
        maybe_examples = self.pre_prompt(instruction=self.instruction, answers=self.answers, thoughts=self.thoughts)
        usr_msgs = [
            (maybe_examples + self.setup.format(text) + self.instruction, VerifyValidClass(extract_ypred_fn=self.extract_ypred_from_json, valid_classes=self.valid_classes))
        ]
        return self.system_prompt, usr_msgs

class ExtractClassifyStrategy(PromptStrategy):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.system_prompt = "You're a helpful expert on the WZ08, the taxonomy of the industrial sectors of Germany."
        
        self.setup = """Given is the following jobposting:\n{}"""
        self.instructions = [
        "\nPlease extract any important hints about the Industry of the company. Answer in json using keys 'summary' (SHORT 2-sentence description) 'company_name' and 'industry'",
        f"\nBased on the previous extraction and the jobposting, please classify the WZ08 industrial section OF THE COMPANY. Answer in json using keys 'industrial_section' (e.g. Vermittlung von Arbeitskräften) and 'reasoning'."
        ]

        self.answers_1 = {
            'example1' : """
            Answer:
            {
                "summary" : "hagebau Handelsgesellschaft für Baustoffe mbH & Co. KG is a leading cooperative in the building materials, wood, 
            flooring, and DIY sectors with over 1,500 locations across Germany, Austria, Switzerland, France, Luxembourg, Belgium, and
            Spain. The company has approximately 1,400 employees and offers various services through its subsidiaries, including 
            insurance, logistics, consulting, and more."
                "company_name" : "hagebau Handelsgesellschaft für Baustoffe mbH & Co. KG",
                "industry" : "Building materials, wood and tile trade",
                
            }""",

            'example2' : """
            Answer:
            {
                "summary" : "The ELSTA GmbH & Co. KG company is looking for experienced roofers/ carpenters (m/w) to be hired immediately, with a focus
            on placements in southern Germany. The job involves assisting with tasks such as roofing, renovation, and maintenance 
            work, requiring physical fitness, teamwork, and independent working abilities."
                "company_name" : "ELSTA GmbH & Co. KG",
                "industry" : "Labor recruitment",
            }""",

            'example3' : """
            Answer:
            {
                "summary" : "The Koch Dachtechnik GmbH is one of the leading companies for waterproofing, trapezoidal sheet and facade work, with 
            locations in Saxony, Saxony-Anhalt, Lower Saxony, Thuringia, and Rhineland-Palatinate. The company was founded in 1878 and
            has a medium-sized enterprise with a strong reputation in the industry."
                "company_name" : "Koch Dachtechnik GmbH",
                "industry" : "Roofing",
            }"""
        }
        self.answers_2 = {
            'example1' : """
            Answer:
            {
                "industrial_section" : "Handel; Instandhaltung und Reperatur von Kraftfahrzeugen",
                "reasoning" : "The employer, hagebau Handelsgesellschaft für Baustoffe mbH & Co. KG is a leading company in the industrial section for 'Baustoff-, Holz- und Fliesenhandel' which matches the description of the industrial section G",
            }
            """,
            'example2' : """
            Answer:
            {
                "industrial_section" : "Vermittlung und Überlassung von Arbeitskräften",
                "reasoning" : "The employer, ELSTA GmbH & Co. KG, is involved in the Vermittlung von Arbeitskräften (placement of workers) industry, which falls under the broader category of 'Vermittlung und Überlassung von Arbeitskräften'.",
            }
            """,
            'example3' : 
            """
            Answer:
            {
                "industrial_section" : "Baugewerbe",
                "reasoning" : "The employer, Koch Dachtechnik GmbH, is involved in 'Dachdeckerei und Bauspenglerei' which matches section F.",
            }
            """

        }
        self.thoughts = None

    def get_messages(self, text : str,  label : str = None):
        maybe_examples1 = self.pre_prompt(self.instructions[0], self.answers_1, self.thoughts)
        maybe_examples2 = self.pre_prompt(self.instructions[1], self.answers_2, self.thoughts)

        usr_msgs = [
            (maybe_examples1 + self.setup.format(text) + self.instructions[0], VerifyJsonOutput()),
            (maybe_examples2 + self.instructions[1], VerifyJsonOutput()),
            (
                f"Based on the extracted information about the industry section above, please use the following map from section_letter to industrial_section: \n{self.mappings}\n" \
            + f"ONLY OUTPUT JSON, using the keys 'section_letter' (i.e. 'A' or 'N'), 'industrial_section' (i.e. Bergbau und Gewinnung von Steinen und Erden) and 'reasoning'",
            VerifyValidClass(self.extract_ypred_from_json, valid_classes=self.valid_classes)
            )
        ]

        return self.system_prompt, usr_msgs

    
        
class ExtractClassifyReflectStrategy(ExtractClassifyStrategy):

    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.reflect_prompt = """
        Please reflect on your answer and if necessary correct it. Answer in the same style as before.
        """

    def get_messages(self, text: str, label: str):
        system_msg, usr_msgs = super().get_messages(text, label)
        manager_obj = copy.deepcopy(usr_msgs[-1][1])
        usr_msgs.append((self.reflect_prompt, manager_obj))

        return system_msg, usr_msgs
    
class ExtractClassifyHypotheses(PromptStrategy):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.system_prompt = "You're a helpful expert on the WZ08, the taxonomy of the industrial sectors of Germany."
        
        self.setup = """Given is the following jobposting:\n{}"""
        self.instructions = [
        "\nPlease extract any important hints about the Industry of the company. Answer in json using keys 'summary' (SHORT 2-sentence description) 'company_name' and 'industry'",
        f"\nBased on the previous extraction and the jobposting, please classify the WZ08 industrial section OF THE COMPANY. Answer in json using keys 'industrial_section' (e.g. Vermittlung von Arbeitskräften) and 'reasoning'.",
        ]

        self.answers_1 = {
            'example1' : """
            Answer:
            {
                "summary" : "hagebau Handelsgesellschaft für Baustoffe mbH & Co. KG is a leading cooperative in the building materials, wood, 
            flooring, and DIY sectors with over 1,500 locations across Germany, Austria, Switzerland, France, Luxembourg, Belgium, and
            Spain. The company has approximately 1,400 employees and offers various services through its subsidiaries, including 
            insurance, logistics, consulting, and more."
                "company_name" : "hagebau Handelsgesellschaft für Baustoffe mbH & Co. KG",
                "industry" : "Building materials, wood and tile trade",
                
            }""",

            'example2' : """
            Answer:
            {
                "summary" : "The ELSTA GmbH & Co. KG company is looking for experienced roofers/ carpenters (m/w) to be hired immediately, with a focus
            on placements in southern Germany. The job involves assisting with tasks such as roofing, renovation, and maintenance 
            work, requiring physical fitness, teamwork, and independent working abilities."
                "company_name" : "ELSTA GmbH & Co. KG",
                "industry" : "Labor recruitment",
            }""",

            'example3' : """
            Answer:
            {
                "summary" : "The Koch Dachtechnik GmbH is one of the leading companies for waterproofing, trapezoidal sheet and facade work, with 
            locations in Saxony, Saxony-Anhalt, Lower Saxony, Thuringia, and Rhineland-Palatinate. The company was founded in 1878 and
            has a medium-sized enterprise with a strong reputation in the industry."
                "company_name" : "Koch Dachtechnik GmbH",
                "industry" : "Roofing",
            }"""
        }
        self.answers_2 = {
            'example1' : """
            Answer:
            {
                "industrial_section" : "Handel; Instandhaltung und Reperatur von Kraftfahrzeugen",
                "reasoning" : "The employer, hagebau Handelsgesellschaft für Baustoffe mbH & Co. KG is a leading company in the industrial section for 'Baustoff-, Holz- und Fliesenhandel' which matches the description of the industrial section G",
            }
            """,
            'example2' : """
            Answer:
            {
                "industrial_section" : "Vermittlung und Überlassung von Arbeitskräften",
                "reasoning" : "The employer, ELSTA GmbH & Co. KG, is involved in the Vermittlung von Arbeitskräften (placement of workers) industry, which falls under the broader category of 'Vermittlung und Überlassung von Arbeitskräften'.",
            }
            """,
            'example3' : 
            """
            Answer:
            {
                "industrial_section" : "Baugewerbe",
                "reasoning" : "The employer, Koch Dachtechnik GmbH, is involved in 'Dachdeckerei und Bauspenglerei' which matches section F.",
            }
            """

        }
        self.thoughts = None

    def get_messages(self, text : str,  label : str = None):
        maybe_examples1 = self.pre_prompt(self.instructions[0], self.answers_1, self.thoughts)
        maybe_examples2 = self.pre_prompt(self.instructions[1], self.answers_2, self.thoughts)

        usr_msgs = [
            (maybe_examples1 + self.setup.format(text) + self.instructions[0], VerifyJsonOutput()),
            (maybe_examples2 + self.instructions[1], GatherHypotheses(n = 5, verify_json=True)),
            (
                f"Based on the given hypothesis about the industry section above, please use the following map from section_letter to industrial_section: \n{self.mappings}\n, retrieve the SINGLE MOST LIKELY HYPOTHESIS." \
            + f"ONLY OUTPUT JSON, using the keys 'section_letter' (i.e. 'A' or 'N'), 'industrial_section' (i.e. Bergbau und Gewinnung von Steinen und Erden) and 'reasoning'",
            VerifyValidClass(self.extract_ypred_from_json, valid_classes=self.valid_classes)
            )
        ]

        return self.system_prompt, usr_msgs