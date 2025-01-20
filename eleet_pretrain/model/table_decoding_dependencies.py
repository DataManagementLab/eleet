import torch
import logging


logger = logging.getLogger(__name__)

    
MAX_ROW_LEN = 500


def get_answer_end(labels, b_id, q_num, answer_start):
    """Get the end of the answer span from the labels."""
    i = answer_start.clone()
    answer_end = torch.zeros_like(answer_start, dtype=answer_start.dtype, device=answer_start.device)
    done = torch.zeros_like(answer_start, dtype=bool, device=answer_start.device)
    while (~done).any():
        i += 1
        overflow = i >= labels.shape[-1]
        answer_end[overflow] = labels.shape[-1]
        done = done | overflow

        span_end = torch.zeros_like(done)
        span_end[~done] = labels[b_id, q_num, i][~done] != 1
        answer_end[span_end] = i[span_end]
        done = done | span_end

    return answer_end
