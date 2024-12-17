from typing import Dict, List, Any, ClassVar, Optional
from pydantic import Field
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI

from .compass import Compass

from .logger import Logger
LOGGER = Logger.create(__name__)

class TestCaseAgent(Chain):
    """
    A chain that generates high-level test cases for a given feature, using information
    from a Compass object that contains enriched method summaries and code.

    This agent:
    - Takes a feature name as input, along with a feature-to-method mapping and a Compass instance.
    - Uses the Compass object to retrieve code snippets associated with the specified feature.
    - Infers the feature's intended behavior from these snippets.
    - Invokes an LLM to produce user-focused test cases without directly referencing the code.
    - Ensures test cases cover positive and negative scenarios, provide clear steps and expected outcomes,
      and are written at a user/tester-friendly abstraction level.

    Attributes:
        feature_dict (Dict[str, List[str]]): Maps feature names to lists of associated method identifiers.
        compass (Compass): A Compass object that contains `method_summaries`, each with 'summary' and 'code' keys.
        model (ChatOpenAI): The language model used to generate test cases.

    Class Attributes:
        input_keys (List[str]): The chain expects one input key, "feature_name".
        output_keys (List[str]): The chain outputs one key, "test_cases".
    """

    feature_dict: Dict[str, List[str]] = Field(
        ...,
        description="Maps feature names to associated method identifiers."
    )
    compass: Compass = Field(
        ...,
        description="A Compass object containing method summaries (including code) for methods."
    )
    model: ChatOpenAI = Field(
        default_factory=lambda: ChatOpenAI(model_name="gpt-4o-mini", temperature=0), # Test case generation model will go here
        description="The LLM used to generate the test cases."
    )

    input_keys: ClassVar[List[str]] = ["feature_dict"]
    output_keys: ClassVar[List[str]] = ["test_cases"]

    @classmethod
    def as_chain(cls, feature_dict: Dict[str, List[str]], compass: Compass, model: Optional[ChatOpenAI] = None) -> "TestCaseAgent":
        """
        Class method to create a TestCaseAgent chain instance.

        Args:
            feature_dict (Dict[str, List[str]]): Mapping of features to methods.
            compass (Compass): A Compass instance with method summaries and code.
            model (Optional[ChatOpenAI]): A custom LLM instance to use. If not provided, a default model is used.

        Returns:
            TestCaseAgent: An instance of the TestCaseAgent chain.
        """
        LOGGER.info("Created test case generation chain.")

        return cls(
            feature_dict=feature_dict,
            compass=compass,
            model=model or ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        )

    def _get_summary_for_feature(self, feature_name: str) -> List[str]:
        """
        Retrieve code snippets associated with a given feature using the Compass object.

        Args:
            feature_name (str): The name of the feature for which to retrieve code.

        Returns:
            List[str]: A list of code snippet strings related to the feature. Returns an empty list if no code is found.
        """
        if feature_name not in self.feature_dict:
            return []

        funcs = self.feature_dict[feature_name]
        snippets = [
            self.compass.method_summaries[m]["summary"]
            for m in funcs
            if m in self.compass.method_summaries and "code" in self.compass.method_summaries[m]
        ]
        print(snippets)
        return snippets
    
    def _generate_test_cases_for_feature(self, feature_name: str, snippets: List[str]) -> str:
        """
        Generate test cases for a single feature given its code snippets.

        Args:
            feature_name (str): The name of the feature.
            snippets (List[str]): Code snippets for this feature.

        Returns:
            str: A string containing the generated test cases for this feature.
                 If no snippets are provided, returns a message indicating no code found.
        """
        LOGGER.debug(f"Generating test cases for given feature: {feature_name}")

        if not snippets:
            return "No code found for this feature."

        test_prompt = f"""
                You are a QA engineer responsible for writing high-level test cases for the feature named '{feature_name}'.
                Below are code snippets related to this feature (for your reference only; do not mention the code or its details in the test cases):

                {'\n'.join(snippets)}

                Using the functionality inferred from the code (e.g., what the feature is intended to do, how it should behave),
                write a comprehensive set of test cases. Each test case should:

                - Not explicitly mention or reference the code.
                - Focus on user-visible behavior, inputs, and expected outcomes.
                - Include a sequence of high-level steps that a tester would follow.
                - Cover both positive and negative scenarios.
                - Be written so that a human QA tester could follow them without knowing the code.

                Provide these test cases in a structured format, for example:

                Test Case Title:
                    Steps:
                    1. ...
                    2. ...
                    Expected Outcome:

                Generate several such test cases to thoroughly cover the feature.
                """.strip()

        response = self.model.invoke(test_prompt)
        return response.content.strip()

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Generate high-level test cases for all features in the feature_dict.

        The method:
        - Extracts the feature_dict from inputs.
        - Iterates over each feature in the dictionary.
        - Retrieves associated code snippets from Compass.
        - Uses the LLM to generate user-focused test cases.
        - Aggregates all test cases into a single dictionary.

        Args:
            inputs (dict): A dictionary containing "feature_dict".

        Returns:
            dict: A dictionary with the key "test_cases", mapping each feature_name to its generated test cases.
        """
        LOGGER.info("Starting test case agent chain.")

        self.feature_dict = inputs["feature_dict"]
        all_test_cases = {}

        for feature_name in self.feature_dict.keys():
            snippets = self._get_summary_for_feature(feature_name)
            test_cases = self._generate_test_cases_for_feature(feature_name, snippets)
            all_test_cases[feature_name] = test_cases

        return {"test_cases": all_test_cases}