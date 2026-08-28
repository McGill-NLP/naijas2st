"""Subprocess worker that extracts SeamlessM4T units from a single audio sample.

Designed to be spawned by ``driver.py`` so that each invocation loads
the model fresh in its own process (avoids CUDA/CPU resident-memory
buildup across the whole dataset). Reads a JSON ``{"audio": ...}``
payload from stdin and prints ``{"units": [...]}`` to stdout.
"""

import json
import sys
import torch
import torchaudio
from transformers import SeamlessM4TProcessor, SeamlessM4TModel

MODEL = "facebook/seamless-m4t-v2-large"
SR = 16000
MAX_SEC = 30
MAX_SAMPLES = SR * MAX_SEC
DEVICE = "cpu"

def chunk(wav):
    """Yield ``MAX_SAMPLES``-long slices of a ``[1, T]`` audio tensor.

    Args:
        wav (torch.Tensor): Mono audio tensor of shape ``[1, T]``.

    Yields:
        torch.Tensor: Consecutive ``[1, K]`` slices where
        ``K <= MAX_SAMPLES``.
    """
    for i in range(0, wav.shape[-1], MAX_SAMPLES):
        yield wav[:, i:i + MAX_SAMPLES]

@torch.no_grad()
def main():
    """Extract SeamlessM4T speech units from one audio sample, JSON-stdin.

    Workflow:
        1. Read a JSON ``{"audio": {"array": [...], "sampling_rate": N}}``
           payload from stdin.
        2. Build a torch tensor and resample to ``SR`` (16 kHz) if
           needed.
        3. Load ``SeamlessM4Tv2Model`` + processor on ``DEVICE`` and
           drop the unused text/T2U components to reduce RAM
           footprint.
        4. Split the waveform into ``MAX_SAMPLES``-long chunks
           (``MAX_SEC = 30`` s each) via :func:`chunk`, feed each
           chunk through the speech encoder, and take ``argmax`` over
           the final hidden state to produce per-frame unit IDs.
        5. Concatenate chunk results and print
           ``{"units": [int, ...]}`` to stdout for the driver to pick
           up.

    Inputs:
        Single audio dict on stdin.

    Outputs:
        A JSON line with ``units`` on stdout.

    Returns:
        None.
    """
    payload = json.loads(sys.stdin.read())
    audio = payload["audio"]

    wav = torch.tensor(audio["array"]).unsqueeze(0)
    sr = audio["sampling_rate"]

    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    processor = SeamlessM4TProcessor.from_pretrained(MODEL)
    model = SeamlessM4TModel.from_pretrained(MODEL).to(DEVICE)
    model.eval()

    # Disable unused components
    model.text_encoder = None
    model.text_decoder = None
    model.t2u_model = None

    units = []

    for c in chunk(wav):
        inputs = processor(
            audio=c.squeeze(0),
            sampling_rate=SR,
            return_tensors="pt"
        )
        enc = model.speech_encoder(inputs.input_features).last_hidden_state
        units.append(enc.argmax(dim=-1).squeeze(0).cpu())

    out = {
        "units": torch.cat(units).tolist()
    }

    print(json.dumps(out))

if __name__ == "__main__":
    main()