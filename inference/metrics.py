from sacrebleu.metrics import CHRF
import textdistance
import numpy as np
from tqdm import tqdm
import concurrent.futures

class SampledLevelEvaluator:
    def evaluate_level_str(self, level_str: str, reference_str: str, metrics: list = None) -> dict:
        """
        Args:
            level_str (str): The flattened string representation of the sampled level.
            reference_str (str): The flattened string for the reference level.
            metrics (list): List of metric keys to compute. Options include:
                "chrF_score", "levenshtein_distance", "hamming_distance",
                "lcs_substring_similarity", "lcs_subsequence_similarity".
                If None, all are computed.

        Returns:
            dict: A dictionary containing scores for the selected metrics.
        """
        if metrics is None:
            metrics = [
                "chrF_score", "levenshtein_distance", "hamming_distance",
                "lcs_substring_similarity", "lcs_subsequence_similarity"
            ]

        results = {}

        if "chrF_score" in metrics:
            chrf = CHRF(word_order=0, char_order=70)
            chrf_score_obj = chrf.corpus_score([level_str], [[reference_str]])
            results["chrF_score"] = chrf_score_obj.score

        if "levenshtein_distance" in metrics:
            results["levenshtein_distance"] = textdistance.levenshtein.distance(level_str, reference_str)
            results["levenshtein_normalized"] = textdistance.levenshtein.normalized_similarity(level_str, reference_str)

        if "hamming_distance" in metrics:
            if len(level_str) == len(reference_str):
                results["hamming_distance"] = textdistance.hamming.distance(level_str, reference_str)
                results["hamming_normalized"] = textdistance.hamming.normalized_similarity(level_str, reference_str)
            else:
                results["hamming_distance"] = None
                results["hamming_normalized"] = None

        if "lcs_substring_similarity" in metrics:
            lcs_str = textdistance.LCSStr()
            results["lcs_substring_similarity"] = lcs_str.similarity(level_str, reference_str)
            results["lcs_substring_normalized"] = lcs_str.normalized_similarity(level_str, reference_str)

        if "lcs_subsequence_similarity" in metrics:
            lcs_seq = textdistance.LCSSeq()
            results["lcs_subsequence_similarity"] = lcs_seq.similarity(level_str, reference_str)
            results["lcs_subsequence_normalized"] = lcs_seq.normalized_similarity(level_str, reference_str)

        return results


    def evaluate_sample_on_dataset(self, sampled_levels: list, reference_str: str, metrics: list = None, max_workers: int = 6) -> dict:
        """
        Evaluate levels on a complete dataset.

        Returns the best level per metric (max for high, min for low).
        """
        if metrics is None:
            metrics = [
                "chrF_score", "levenshtein_distance", "hamming_distance",
                "lcs_substring_similarity", "lcs_subsequence_similarity"
            ]

        use_parallel = ("lcs_substring_similarity" in metrics or "lcs_subsequence_similarity" in metrics)
        higher_better = {"chrF_score", "lcs_substring_similarity", "lcs_subsequence_similarity", "hamming_normalized"}
        lower_better = {"levenshtein_distance", "hamming_distance"}

        best_metrics = {}
        for metric in metrics:
            if metric in higher_better:
                best_metrics[metric] = {"score": -float("inf"), "level": None}
            elif metric in lower_better:
                best_metrics[metric] = {"score": float("inf"), "level": None}
            else:
                raise ValueError(f"Issue with metrics, no specification on higher or lower better")

        all_sample_metrics = []
        if use_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_sample = {
                    executor.submit(self.evaluate_level_str, sample, reference_str, metrics): sample
                    for sample in sampled_levels
                }

                for future in tqdm(concurrent.futures.as_completed(future_to_sample),
                                   total=len(sampled_levels),
                                   desc="Calculating metrics (parallel)"):
                    sample = future_to_sample[future]
                    try:
                        calculated_metrics = future.result()
                        all_sample_metrics.append({"level": sample, "metrics": calculated_metrics})
                    except Exception as exc:
                        print(f'Parallel sample generation for level "{sample[:20]}..." threw an error: {exc}')
        else:
            for sample in tqdm(sampled_levels, desc="Calculating metrics (sequential)"):
                calculated_metrics = self.evaluate_level_str(sample, reference_str, metrics=metrics)
                all_sample_metrics.append({"level": sample, "metrics": calculated_metrics})


        for result in tqdm(all_sample_metrics, desc="Finding best metrics"):
            level_str = result["level"]
            sample_metrics = result["metrics"]
            for metric in metrics:
                if metric not in sample_metrics:
                    raise ValueError(f"Metric {metric} not found in sample metrics")
                value = sample_metrics.get(metric)

                if metric in higher_better:
                    if value is not None and value > best_metrics[metric]["score"]:
                        best_metrics[metric]["score"] = value
                        best_metrics[metric]["level"] = level_str
                elif metric in lower_better:
                    if value is not None and value < best_metrics[metric]["score"]:
                        best_metrics[metric]["score"] = value
                        best_metrics[metric]["level"] = level_str

        return best_metrics