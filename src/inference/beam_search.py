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
        Perform beam search decoding (sequential, one batch item at a time).
        
        Args:
            src: Source sequence (batch, src_len)
            
        Returns:
            List of best decoded sequences for each batch item
        """
        self.model.eval()
        batch_size = src.size(0)
        
        # Process each batch item sequentially
        results = []
        for i in range(batch_size):
            result = self._beam_search_single(src[i:i+1])
            results.append(result)
        
        return results
    
    def _beam_search_single(self, src: torch.Tensor) -> List[int]:
        """
        Beam search for a single source sequence.
        
        Args:
            src: Single source sequence (1, src_len)
            
        Returns:
            Best decoded sequence (list of token IDs)
        """
        device = src.device
        
        # Encode source
        memory = self.model.encode(src)  # (1, src_len, d_model)
        
        # Expand memory for beam_size
        # (1, src_len, d_model) -> (beam_size, src_len, d_model)
        memory = memory.repeat(self.beam_size, 1, 1)
        
        # Initialize beams with BOS token
        # Each beam: [BOS]
        beams = [[self.bos_idx] for _ in range(self.beam_size)]
        beam_scores = [0.0] + [-1e9] * (self.beam_size - 1)  # Only first beam is active
        finished_beams = []
        
        for step in range(self.max_length):
            # Prepare decoder input for all beams
            # (beam_size, current_length)
            decoder_input = torch.tensor(beams, dtype=torch.long, device=device)

            # Decode: (beam_size, current_length, vocab_size)
            logits = self.model.decode(decoder_input, memory)

            # Get log probs for next token: (beam_size, vocab_size)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            # Collect all candidates
            candidates = []

            for beam_idx in range(len(beams)):
                if beam_scores[beam_idx] == -1e9:  # Inactive beam
                    continue

                # Get top-k tokens for this beam
                top_log_probs, top_indices = log_probs[beam_idx].topk(self.beam_size)

                for log_prob, token_id in zip(top_log_probs.tolist(), top_indices.tolist()):
                    # New sequence
                    new_seq = beams[beam_idx] + [token_id]
                    # New score
                    new_score = beam_scores[beam_idx] + log_prob

                    # If EOS, add to finished beams
                    if token_id == self.eos_idx:
                        # Apply length penalty
                        seq_length = len(new_seq) - 1  # Exclude BOS
                        normalized_score = self._length_normalize(new_score, seq_length)
                        finished_beams.append((new_seq, normalized_score))
                    else:
                        candidates.append((new_seq, new_score))
            
            # If no candidates left, break
            if not candidates and not beams:
                break
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:self.beam_size]
            
            # Update beams
            if candidates:
                beams = [seq for seq, _ in candidates]
                beam_scores = [score for _, score in candidates]
            else:
                # No more active beams
                break
            
            # Early stopping if we have enough finished beams
            if len(finished_beams) >= self.beam_size and self.early_stopping:
                break
        
        # Add remaining beams to finished beams (with length penalty)
        for beam, score in zip(beams, beam_scores):
            if score != -1e9:
                seq_length = len(beam) - 1  # Exclude BOS
                normalized_score = self._length_normalize(score, seq_length)
                finished_beams.append((beam, normalized_score))
        
        # Select best beam
        if finished_beams:
            finished_beams.sort(key=lambda x: x[1], reverse=True)
            best_seq = finished_beams[0][0]
        else:
            # Fallback: use first beam
            best_seq = beams[0] if beams else [self.bos_idx, self.eos_idx]
        
        return best_seq


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
