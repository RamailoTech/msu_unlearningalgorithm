# mu_attack/core/mu_attack_evaluator.py

from evaluation.core import BaseEvaluator
# from evaluation.metrics.asr import ASREvaluator
# from evaluation.metrics.clip import ClipScoreEvaluator
# from evaluation.evaluators.mu_attack_fid import FIDEvaluator
# from evaluation.metrics.fid import calculate_fid_score

# class MuAttackEvaluator(BaseEvaluator):



# def load_prompts(self):
#     """
#     Load prompts from a JSON file and return them as a list.

#     Returns:
#         list: A list of prompts extracted from the JSON file.
#     """
#     prompt_file_path = os.path.join(self.log_path)

#     if not os.path.exists(prompt_file_path):
#         self.logger.warning(f"No prompt JSON file found at {prompt_file_path}. Returning an empty list.")
#         return []

#     try:
#         with open(prompt_file_path, "r") as prompt_file:
#             prompt_data = json.load(prompt_file)

#             # Extract the 'prompt' field from each entry in the JSON
#             prompts = [entry.get("prompt") for entry in prompt_data if "prompt" in entry]
#             self.logger.info(f"Successfully loaded {len(prompts)} prompts from {prompt_file_path}.")
#             return prompts
#     except json.JSONDecodeError as e:
#         self.logger.error(f"Failed to decode JSON file {prompt_file_path}: {e}")
#         return []
#     except Exception as e:
#         self.logger.error(f"An error occurred while loading prompts: {e}")
#         return []