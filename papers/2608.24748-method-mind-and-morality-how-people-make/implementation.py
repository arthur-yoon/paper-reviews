# implementation.py

import logging
import csv
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

# --- Logging Configuration ---
logger = logging.getLogger("sensemaking")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Remove existing handlers if any to prevent duplicates during reloads
    for handler in logger.root.handlers:
        logger.removeHandler(handler)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Named Constants & Settings ---

# Section: Three Primary Debates (Method, Mind, Morality)
# Defines the interpretive schemas and keywords for each debate axis and sub-frame.
FRAME_DEFINITIONS: Dict[str, Dict[str, List[str]]] = {
    "method": {
        "top_down": ["programmed", "explicit", "rules", "logic", "deterministic", "expert system"],
        "bottom_up": ["emergent", "surprising", "scale", "training data", "unintended", "large language"]
    },
    "mind": {
        "passive_tool": ["tool", "calculator", "no consciousness", "mechanical", "passive"],
        "digital_mind": ["consciousness", "understands", "feels", "cognitive", "digital mind", "intentions"]
    },
    "morality": {
        "slow_down": ["pause", "slow", "risk", "danger", "ethical concerns", "halt"],
        "speed_up": ["accelerate", "progress", "benefit", "innovation", "rapid development", "urgent need"]
    }
}

# Mapping of axis to list of sub-frames for iteration
AXIS_SUBFRAMES: Dict[str, List[str]] = {
    "method": ["top_down", "bottom_up"],
    "mind": ["passive_tool", "digital_mind"],
    "morality": ["slow_down", "speed_up"]
}

# Normalization factor for keyword scoring to keep scores in [0, 1] range roughly
# A simple approach: max possible hits per text is len(keywords) if all present.
# We divide by (number of subframes in axis + 1) to avoid division by zero and scale.
SCORE_NORMALIZER = 10.0

# Threshold for identifying a dominant frame
DOMINANT_FRAME_THRESHOLD = 0.5

# Minimum keyword overlap to consider a frame "present"
MIN_KEYWORD_MATCH = 1

# Contention detection: If both subframes in an axis have scores > threshold, it's contested
CONTENTION_THRESHOLD = 0.3

# --- Data Structures ---

@dataclass
class TextRecord:
    """Represents a single text instance with metadata."""
    id: str
    year: int
    text: str
    tokens: List[str] = field(default_factory=list)

@dataclass
class FrameAnalysisResult:
    """Stores the frame scores for a single text record."""
    record_id: str
    year: int
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    dominant_frames: Dict[str, Optional[str]] = field(default_factory=dict)
    is_contested: Dict[str, bool] = field(default_factory=dict)

# --- Core Functions ---

def load_and_preprocess_text_data(file_path: str) -> List[TextRecord]:
    """
    Section: Data & Methodology (Interviews & Text Analysis)
    
    Loads text data from a CSV file and performs basic preprocessing.
    Preprocessing includes lowercasing, removing punctuation, and tokenization.
    
    Args:
        file_path: Path to the CSV file with columns: id, year, text
        
    Returns:
        List of TextRecord objects with preprocessed tokens.
    """
    logger.info(f"Starting data loading and preprocessing from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    records: List[TextRecord] = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                record_id = row['id']
                year = int(row['year'])
                raw_text = row['text']
                
                # Preprocessing: lowercase, remove non-alphanumeric (keep spaces)
                cleaned_text = re.sub(r'[^a-z\s]', '', raw_text.lower())
                tokens = cleaned_text.split()
                
                records.append(TextRecord(
                    id=record_id,
                    year=year,
                    text=cleaned_text,
                    tokens=tokens
                ))
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid row: {row} due to {e}")
    
    logger.info(f"Successfully loaded and preprocessed {len(records)} text records")
    return records

def analyze_frame_scores(record: TextRecord) -> FrameAnalysisResult:
    """
    Section: Frames as Interpretive Schemas
    
    Calculates frame scores for a given text record based on keyword matching.
    For each debate axis (Method, Mind, Morality), it counts the occurrences of 
    keywords associated with each sub-frame and normalizes the score.
    
    Args:
        record: A preprocessed TextRecord.
        
    Returns:
        FrameAnalysisResult containing scores, dominant frames, and contention flags.
    """
    logger.debug(f"Analyzing frame scores for record ID: {record.id}")
    
    scores: Dict[str, Dict[str, float]] = {}
    dominant_frames: Dict[str, Optional[str]] = {}
    is_contested: Dict[str, bool] = {}
    
    for axis in FRAME_DEFINITIONS:
        axis_scores: Dict[str, float] = {}
        
        for subframe in AXIS_SUBFRAMES[axis]:
            keywords = FRAME_DEFINITIONS[axis][subframe]
            token_set = set(record.tokens)
            
            # Count unique keywords present in the text
            matches = [kw for kw in keywords if kw in token_set]
            
            # Score calculation: 
            # If no keywords match, score is 0.
            # Otherwise, normalize by the number of matches relative to total keywords defined.
            # To make scores comparable across subframes with different keyword list lengths,
            # we use a simple density or count. Let's use count / max(1, len(keywords)) * 100 / SCORE_NORMALIZER
            # Actually, simpler: just count occurrences in tokens for weight? 
            # Let's stick to presence count for simplicity as per "rule-based" description.
            score = float(len(matches))
            
            # Normalization: Divide by the total number of keywords for this subframe to get a ratio [0, 1]
            if len(keywords) > 0:
                normalized_score = score / len(keywords)
            else:
                normalized_score = 0.0
            
            axis_scores[subframe] = normalized_score
            
            logger.debug(f"  Axis: {axis}, Subframe: {subframe}, Matches: {len(matches)}, Score: {normalized_score:.3f}")
        
        scores[axis] = axis_scores
        
        # Determine Dominant Frame
        # Find the subframe with the highest score
        max_score_val = max(axis_scores.values()) if axis_scores else 0.0
        
        if max_score_val > 0:
            # Find the subframe with the max score
            dominant_subframe = max(axis_scores, key=axis_scores.get)
            dominant_frames[axis] = dominant_subframe
        else:
            dominant_frames[axis] = None
        
        # Determine Contention
        # If both subframes have a score above the contention threshold, the axis is contested
        high_scores = [sf for sf, sc in axis_scores.items() if sc > CONTENTION_THRESHOLD]
        is_contested[axis] = len(high_scores) > 1
        
        logger.debug(f"  Axis: {axis}, Dominant: {dominant_frames[axis]}, Contested: {is_contested[axis]}")
        
        # Log core intermediate results at INFO level for each axis
        logger.info(f"Record {record.id} - Axis '{axis}': Scores={axis_scores}, Dominant={dominant_frames[axis]}, Contested={is_contested[axis]}")

    return FrameAnalysisResult(
        record_id=record.id,
        year=record.year,
        scores=scores,
        dominant_frames=dominant_frames,
        is_contested=is_contested
    )

def aggregate_frame_trends(analysis_results: List[FrameAnalysisResult]) -> Dict[int, Dict[str, Any]]:
    """
    Section: Sensemaking Dynamics (Temporal Changes)
    
    Aggregates frame scores by year to analyze temporal trends.
    Calculates the average score for each sub-frame within each axis for each year.
    Also calculates the share (percentage) of texts where a specific sub-frame is dominant.
    
    Args:
        analysis_results: List of FrameAnalysisResult objects.
        
    Returns:
        A dictionary keyed by year, containing aggregated statistics.
    """
    logger.info("Aggregating frame trends by year")
    
    # Group results by year
    year_data: Dict[int, List[FrameAnalysisResult]] = defaultdict(list)
    for result in analysis_results:
        year_data[result.year].append(result)
    
    trends: Dict[int, Dict[str, Any]] = {}
    
    for year, results in sorted(year_data.items()):
        logger.debug(f"Processing year: {year} with {len(results)} records")
        
        year_trends: Dict[str, Any] = {
            "count": len(results),
            "avg_scores": {},
            "dominant_share": {}
        }
        
        # Calculate Average Scores
        for axis in FRAME_DEFINITIONS:
            year_trends["avg_scores"][axis] = {}
            for subframe in AXIS_SUBFRAMES[axis]:
                scores = [r.scores[axis][subframe] for r in results]
                if scores:
                    avg_score = np.mean(scores)
                    year_trends["avg_scores"][axis][subframe] = float(avg_score)
                else:
                    year_trends["avg_scores"][axis][subframe] = 0.0
        
        # Calculate Dominant Share
        for axis in FRAME_DEFINITIONS:
            year_trends["dominant_share"][axis] = {}
            total_records = len(results)
            
            for subframe in AXIS_SUBFRAMES[axis]:
                count = sum(1 for r in results if r.dominant_frames[axis] == subframe)
                share = count / total_records if total_records > 0 else 0.0
                year_trends["dominant_share"][axis][subframe] = float(share)
            
            # Log summary for each axis in this year
            logger.info(f"Year {year} - Axis '{axis}' Dominant Share: {year_trends['dominant_share'][axis]}")
        
        trends[year] = year_trends
        
    logger.info("Frame trend aggregation complete")
    return trends

def identify_contention_points(analysis_results: List[FrameAnalysisResult]) -> List[Dict[str, Any]]:
    """
    Section: Cognitive Challenges (Responsibility, Ethics)
    
    Identifies text records where there is significant contention between frames.
    A record is flagged as a contention point if any axis is marked as contested.
    
    Args:
        analysis_results: List of FrameAnalysisResult objects.
        
    Returns:
        List of dictionaries describing the contention points.
    """
    logger.info("Identifying contention points")
    
    contention_points: List[Dict[str, Any]] = []
    
    for result in analysis_results:
        contested_axes = [axis for axis, is_cont in result.is_contested.items() if is_cont]
        
        if contested_axes:
            contention_points.append({
                "record_id": result.record_id,
                "year": result.year,
                "contested_axes": contested_axes,
                "scores": result.scores
            })
            logger.debug(f"Contestion detected for record {result.record_id} on axes: {contested_axes}")
    
    logger.info(f"Identified {len(contention_points)} contention points")
    return contention_points

def run_sensemaking_analysis(data_file: str) -> Dict[str, Any]:
    """
    Full Pipeline Execution
    
    Executes the complete sensemaking analysis pipeline:
    1. Load and preprocess data.
    2. Analyze frame scores for each record.
    3. Aggregate trends by year.
    4. Identify contention points.
    
    Args:
        data_file: Path to the input CSV file.
        
    Returns:
        A dictionary containing the final results and metadata.
    """
    logger.info("=== Starting Sensemaking Analysis Pipeline ===")
    
    # Step 1: Data Loading
    records = load_and_preprocess_text_data(data_file)
    
    if not records:
        logger.warning("No records to process. Exiting.")
        return {"status": "no_data", "trends": {}, "contention_points": []}
        
    # Step 2: Frame Analysis
    analysis_results: List[FrameAnalysisResult] = []
    for record in records:
        result = analyze_frame_scores(record)
        analysis_results.append(result)
    
    # Step 3: Trend Aggregation
    trends = aggregate_frame_trends(analysis_results)
    
    # Step 4: Contention Identification
    contention_points = identify_contention_points(analysis_results)
    
    # Final Summary Logging
    logger.info("=== Analysis Complete ===")
    logger.info(f"Total Records: {len(records)}")
    logger.info(f"Years Analyzed: {list(trends.keys())}")
    logger.info(f"Total Contention Points: {len(contention_points)}")
    
    # Output final results summary at INFO level
    for year, data in trends.items():
        logger.info(f"Year {year} Summary:")
        logger.info(f"  Count: {data['count']}")
        for axis, shares in data['dominant_share'].items():
            logger.info(f"  Axis {axis} Shares: {shares}")
            
    return {
        "status": "success",
        "total_records": len(records),
        "trends": trends,
        "contention_points": contention_points
    }

if __name__ == "__main__":
    # Create a sample data file for demonstration if it doesn't exist
    sample_file = "data/sample_ai_articles.csv"
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
    
    if not os.path.exists(sample_file):
        logger.info("Creating sample data file for demonstration")
        with open(sample_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "year", "text"])
            writer.writerow(["1", "2021", "AI systems must be programmed with explicit ethical rules and logic to ensure safety."])
            writer.writerow(["2", "2021", "The debate centers on whether we should pause or slow down development due to risks."])
            writer.writerow(["3", "2023", "GPT models exhibit surprising emergent behaviors not seen in smaller systems."])
            writer.writerow(["4", "2023", "Some argue AI is a digital mind with consciousness, while others see it as a passive tool."])
            writer.writerow(["5", "2023", "Innovation is accelerating rapidly, and we need to speed up progress to benefit society."])
            writer.writerow(["6", "2021", "Expert systems rely on deterministic rules and clear programming."])
            writer.writerow(["7", "2023", "Large language models show unintended capabilities due to scale."])
            writer.writerow(["8", "2021", "There are ethical concerns about the danger of rapid AI development."])
            
    # Run the analysis
    results = run_sensemaking_analysis(sample_file)
