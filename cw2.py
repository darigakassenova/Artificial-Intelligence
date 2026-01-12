"""
CW2: Neuro-Symbolic AI System
Student Name: Dariga
Student ID: 250477881

This module implements a neuro-symbolic AI system that combines:
- Computer Vision (CIFAR-100 object recognition)
- Natural Language Processing (Skip-gram word embeddings)
- Symbolic Planning (PDDL planning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Tuple, Optional
from pathlib import Path
import warnings

# lab imports
from lab6 import SkipGramModel
from lab8 import ( 
    ImageEncoder, 
)
from lab9 import (
    Predicate,
    Action,
    State,
    PDDLParser,
    ActionGrounder,
    bfs_search,
    astar_search,
    create_custom_problem,
    validate_user_conditions
)


# ============================================================================
# SECTION 1: CIFAR-100 SEMANTIC EXPANSION
# ============================================================================

# DO NOT CHANGE THIS FUNCTION's signature
def build_my_embeddings(checkpoint_path: str = "best_skipgram_523words.pth") -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load and return your trained Skip-gram embeddings.
    
    This function serves as the entry point for loading your final embedding model
    that contains all Visual Genome words AND all 100 CIFAR-100 classes.
    
    Args:
        checkpoint_path: Path to your saved model checkpoint
        
    Returns:
        vocab: Dictionary mapping words to indices {word: index}
        embeddings: Numpy array of shape (vocab_size, embedding_dim)
        
    Example:
        >>> vocab, embeddings = build_my_embeddings()
        >>> print(f"Vocabulary size: {len(vocab)}")
        >>> print(f"Embedding dimension: {embeddings.shape[1]}")
        >>> print(f"'airplane' index: {vocab.get('airplane', 'NOT FOUND')}")
    """
    # TODO: Implement this function
    # 1. Load your checkpoint file
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # 2. Extract the vocabulary dictionary

    nodes = ckpt["nodes"]
    vocab = {w: i for i, w in enumerate(nodes)}
    
    # 3. Extract the embedding matrix
    embeddings = ckpt["model_state_dict"]["center_embeddings.weight"]
    
    # 4. Ensure vocabulary contains all required words (Visual Genome + CIFAR-100) 
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    embeddings = np.asarray(embeddings, dtype=np.float32)

    assert embeddings.shape[0] == len(vocab), \
        "Mismatch between vocabulary size and embedding matrix"
    
    return vocab, embeddings


# ============================================================================
# SECTION 2: NEURO-SYMBOLIC AI - MULTI-MODAL PLANNING
# ============================================================================

# DO NOT CHANGE THIS FUNCTION's signature
def plan_generator(input_data: Union[torch.Tensor, str],    # ASSUME default CIFAR-100 image dimensions
                  initial_state: List[str],                 # Consistent with Lab9 syntax
                  goal_state: List[str],                    # Consistent with Lab9 syntax
                  domain_file: str = "domain.pddl",
                  skipgram_path: str = "best_skipgram_523words.pth",
                  projection_path: str = "best_cifar100_projection.pth") -> Optional[List[str]]:
    """
    !!!WARNING!!!: Treat this as pseudocode. You may need to modify the logic. 
    
    Main entry point for the neuro-symbolic planning system.
    
    This function implements the complete pipeline from perception to planning.
    
    Args:
        input_data: Either an image tensor OR object name string
        initial_state: List of predicates describing initial state                      
        goal_state: List of predicates describing goal state                   
        domain_file: Path to the PDDL domain file
        skipgram_path: Path to Skip-gram embeddings checkpoint
        projection_path: Path to CIFAR-100 projection model checkpoint
        
    Returns:
        A list of action strings representing the plan, 
            OR None if:
                - The object cannot be identified
                - No valid plan exists
                - ...
        
    Example:
        >>> image = # CIFAR-100 image
        >>> initial = ["on table"]
        >>> goal = ["in basket"]
        >>> plan = plan_generator(image, initial, goal, "domain.pddl")        
    """
    
    # TREAT THIS AS SUGGESTED PSEUDOCODE. YOU MAY USE OTHER PARADIGMS
    # print("STEP 0: start plan_generator")
                    
    # Step 0: Initialize the planner
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_problem = str(Path(domain_file).with_name("problem.pddl"))
    
    try:
        sg = torch.load(skipgram_path, map_location="cpu")
        words = [w.lower() for w in sg["nodes"]]
        E = sg["model_state_dict"]["center_embeddings.weight"].detach().to(device)  # (V,D)
        E = F.normalize(E, dim=1)
    except Exception:
        return None

    # print("STEP: skipgram loaded")

    try:
        proj = torch.load(projection_path, map_location="cpu")
        proj_head = proj["projection_head"]
        proj_dim = proj_head["3.weight"].shape[0] 

        model = ImageEncoder(proj_dim=proj_dim, device=device).to(device)
        model.load_state_dict(proj["model_state_dict"], strict=False)
        model.projection.load_state_dict(proj_head, strict=True)
        model.eval()
    except Exception:
        return None
        
    # print("STEP: projection model loaded")
    
    # Step 1: Identify the object
    obj_name: Optional[str] = None

    if isinstance(input_data, str):
        # Text input
        w = input_data.strip().lower()
        if w not in words:
            return None
        obj_name = w

    else:
        # Image input
        try:
            x = input_data
            if x.dim() == 3:
                x = x.unsqueeze(0) 
            if x.dim() != 4 or x.size(1) != 3:
                return None

            x = x.to(device).float()
            if x.max() > 1.5: 
                x = x / 255.0

            if x.shape[-2:] != (224, 224):
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x = (x - mean) / std

            with torch.no_grad():
                _, z = model(x)  # (1,proj_dim)
                z = F.normalize(z, dim=1)

            sims = torch.matmul(z, E.T).squeeze(0)  # (V,)
            idx = int(torch.argmax(sims).item())
            if idx < 0 or idx >= len(words):
                return None
            obj_name = words[idx]
        except Exception:
            return None

    if obj_name is None:
        return None

    if not isinstance(input_data, str):
        initial_state = [s.replace("?x", obj_name) for s in initial_state]
        goal_state = [s.replace("?x", obj_name) for s in goal_state]
        
    # print("STEP 1: recognized object =", obj_name)
    
    # Step 2: Parse PDDL domain
    init_set = set(initial_state)
    goal_set = set(goal_state)

    try:
        validate_user_conditions(init_set)
        validate_user_conditions(goal_set)
    except Exception:
        return None

    # print("STEP 2: creating PDDL problem")
    
    # Step 3: Create PDDL problem
    try:
        temp_prob = create_custom_problem(base_problem, init_set, goal_set, name=obj_name)
    except Exception:
        return None

    # print("STEP 2.1: PDDL problem created")
    
    # Step 4: Generate plan
    try:
        actions_dict = PDDLParser.parse_domain(domain_file)
        objs, init_state_obj, goal_obj = PDDLParser.parse_problem(temp_prob)

        # print("STEP 3: domain parsed")

        grounder = ActionGrounder(actions_dict, objs)
        grounded_actions = grounder.ground_all()

        # print("STEP 3.2: grounded actions =", len(grounded_actions))

        # print("STEP 4: starting A* search")
        plan_actions = astar_search(init_state_obj, goal_obj, grounded_actions, verbose=False)

        # print("STEP 4.1: A* finished")

        # fallback to BFS if A* fails 
        # if plan_actions is None:
        #     plan_actions = bfs_search(init_state_obj, goal_obj, grounded_actions, verbose=False)

        if plan_actions is None:
            return None

        return [str(a) for a in plan_actions]
    except Exception:
        return None
