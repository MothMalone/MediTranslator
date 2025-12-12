"""
Beam Search Decoding
Advanced decoding strategy that maintains multiple hypotheses at each step.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BeamHypothesis:
    """Single beam hypothesis."""
    tokens: List[int]
    score: float
    is_finished: bool = False


class BeamSearchDecoder:
    """
    Beam Search Decoder for sequence generation.
    
    Maintains K best hypotheses at each decoding step.
    
    Args:
        model: Transformer model
        beam_size: Number of beams
        max_length: Maximum output length
        bos_idx: BOS token index
        eos_idx: EOS token index
        pad_idx: Padding token index
        length_penalty: Length normalization factor (alpha)
        early_stopping: Whether to stop when all beams have finished
    """
    
    def __init__(
        self,
        model: nn.Module,
        beam_size: int = 5,
        max_length: int = 128,
        bos_idx: int = 2,
        eos_idx: int = 3,
        pad_idx: int = 0,
        length_penalty: float = 0.6,
        early_stopping: bool = True
    ):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
    
    def _length_normalize(self, score: float, length: int) -> float:
        """Apply length penalty to score."""
        return score / (length ** self.length_penalty)
    
    @torch.no_grad()
    def decode(self, src: torch.Tensor) -> List[List[int]]:
        """
        Perform beam search decoding.
        
        Args:
            src: Source sequence (batch, src_len)
            
        Returns:
            List of best decoded sequences for each batch item
        """
        self.model.eval()
        device = src.device
        batch_size = src.size(0)
        
        # For simplicity, process one sequence at a time
        results = []
        
        for b in range(batch_size):
            src_single = src[b:b+1]  # (1, src_len)
            best_sequence = self._beam_search_single(src_single)
            results.append(best_sequence)
        
        return results
    
    def _beam_search_single(self, src: torch.Tensor) -> List[int]:
        """
        Beam search for a single source sequence.
        
        Args:
            src: Source sequence (1, src_len)
            
        Returns:
            Best decoded sequence as list of token indices
        """
        device = src.device
        
        # Encode source
        memory = self.model.encode(src)
        
        # Expand memory for beam search
        # (1, src_len, d_model) -> (beam_size, src_len, d_model)
        memory = memory.expand(self.beam_size, -1, -1)
        
        # Initialize beams
        # Each beam: (tokens, cumulative_log_prob)
        beams = [([self.bos_idx], 0.0)]
        finished_beams = []
        
        for step in range(self.max_length):
            if len(beams) == 0:
                break
            
            all_candidates = []
            
            for beam_idx, (tokens, score) in enumerate(beams):
                # Create decoder input
                decoder_input = torch.tensor(
                    [tokens],
                    dtype=torch.long,
                    device=device
                )
                
                # Decode
                logits = self.model.decode(
                    decoder_input,
                    memory[beam_idx:beam_idx+1]
                )
                
                # Get log probabilities for last position
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                
                # Get top-k candidates
                topk_log_probs, topk_indices = log_probs.topk(self.beam_size)
                
                for k in range(self.beam_size):
                    token = topk_indices[k].item()
                    token_score = topk_log_probs[k].item()
                    new_score = score + token_score
                    new_tokens = tokens + [token]
                    
                    if token == self.eos_idx:
                        # Beam finished
                        normalized_score = self._length_normalize(
                            new_score, len(new_tokens)
                        )
                        finished_beams.append((new_tokens, normalized_score))
                    else:
                        all_candidates.append((new_tokens, new_score))
            
            # Select top beams
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:self.beam_size]
            
            # Early stopping
            if self.early_stopping and len(finished_beams) >= self.beam_size:
                break
        
        # Add unfinished beams to finished
        for tokens, score in beams:
            normalized_score = self._length_normalize(score, len(tokens))
            finished_beams.append((tokens, normalized_score))
        
        # Sort by score and return best
        finished_beams.sort(key=lambda x: x[1], reverse=True)
        
        if finished_beams:
            return finished_beams[0][0]
        else:
            return [self.bos_idx, self.eos_idx]


@torch.no_grad()
def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    beam_size: int = 5,
    max_length: int = 128,
    bos_idx: int = 2,
    eos_idx: int = 3,
    pad_idx: int = 0,
    length_penalty: float = 0.6
) -> List[List[int]]:
    """
    Convenience function for beam search decoding.
    
    Args:
        model: Transformer model
        src: Source sequence (batch, src_len)
        beam_size: Number of beams
        max_length: Maximum output length
        bos_idx: BOS token index
        eos_idx: EOS token index
        pad_idx: Padding token index
        length_penalty: Length normalization factor
        
    Returns:
        List of decoded sequences
    """
    decoder = BeamSearchDecoder(
        model=model,
        beam_size=beam_size,
        max_length=max_length,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        length_penalty=length_penalty
    )
    
    return decoder.decode(src)


if __name__ == "__main__":
    print("Beam search module loaded successfully")
